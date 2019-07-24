"""Tests for hf.core.models.bev_rpn"""

import numpy as np
import tensorflow as tf

import hf
import hf.builders.config_builder_util as config_builder
from hf.builders.dataset_builder import DatasetBuilder
from hf.core.models.rpn_model import RpnModel
from hf.protos import pipeline_pb2


class RpnModelTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        config_path = hf.root_dir() + "/configs/rpn_cars.config"
        # Parse pipeline config
        model_config, _, _, dataset_config = config_builder.get_configs_from_pipeline_file(
            config_path, is_training=True
        )

        cls.model_config = model_config
        cls.dataset = DatasetBuilder.build_kitti_dataset(dataset_config, False)

    """
    def test_rpn_loss(self):
        # Use "val" so that the first sample is loaded each time
        rpn_model = RpnModel(
            self.model_config, train_val_test="val", dataset=self.dataset
        )

        predictions = rpn_model.build()

        loss, total_loss = rpn_model.loss(predictions)

        feed_dict = rpn_model.create_feed_dict()

        with self.test_session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            loss_dict_out = sess.run(loss, feed_dict=feed_dict)
            print("Losses ", loss_dict_out)
    """

    def test_gather_mean_sizes(self):
        rpn_model = RpnModel(
            self.model_config, train_val_test="val", dataset=self.dataset
        )
        mean_sizes = rpn_model._gather_mean_sizes(
            tf.convert_to_tensor(
                np.asarray(rpn_model._cluster_sizes, dtype=np.float32)
            ),
            tf.constant([[1, 0, 1], [0, 0, 1]]),
        )
        with self.test_session() as sess:
            mean_sizes = sess.run(mean_sizes)
            print(mean_sizes)

    """
    def test_create_path_drop_masks(self):
        # Tests creating path drop choices
        # based on the given probabilities

        rpn_model = RpnModel(self.model_config,
                             train_val_test="val",
                             dataset=self.dataset)
        rpn_model.build()
        ##################################
        # Test-Case 1 : Keep img, Keep bev
        ##################################
        p_img = tf.constant(0.6)
        p_bev = tf.constant(0.85)

        # Set the random numbers for testing purposes
        rand_choice = [0.53, 0.83, 0.05]
        rand_choice_tensor = tf.convert_to_tensor(rand_choice)

        img_mask, bev_mask = rpn_model.create_path_drop_masks(
            p_img, p_bev, rand_choice_tensor)

        with self.test_session():
            img_mask_out = img_mask.eval()
            bev_mask_out = bev_mask.eval()
            np.testing.assert_array_equal(img_mask_out, 1.0)
            np.testing.assert_array_equal(bev_mask_out, 1.0)

        ##################################
        # Test-Case 2 : Kill img, Keep bev
        ##################################
        p_img = tf.constant(0.2)
        p_bev = tf.constant(0.85)

        img_mask, bev_mask = rpn_model.create_path_drop_masks(
            p_img, p_bev, rand_choice_tensor)

        with self.test_session():
            img_mask_out = img_mask.eval()
            bev_mask_out = bev_mask.eval()
            np.testing.assert_array_equal(img_mask_out, 0.0)
            np.testing.assert_array_equal(bev_mask_out, 1.0)

        ##################################
        # Test-Case 3 : Keep img, Kill bev
        ##################################
        p_img = tf.constant(0.9)
        p_bev = tf.constant(0.1)

        img_mask, bev_mask = rpn_model.create_path_drop_masks(
            p_img, p_bev, rand_choice_tensor)

        with self.test_session():
            img_mask_out = img_mask.eval()
            bev_mask_out = bev_mask.eval()
            np.testing.assert_array_equal(img_mask_out, 1.0)
            np.testing.assert_array_equal(bev_mask_out, 0.0)

        ##############################################
        # Test-Case 4 : Kill img, Kill bev, third flip
        ##############################################
        p_img = tf.constant(0.0)
        p_bev = tf.constant(0.1)

        img_mask, bev_mask = rpn_model.create_path_drop_masks(
            p_img, p_bev, rand_choice_tensor)

        with self.test_session():
            img_mask_out = img_mask.eval()
            bev_mask_out = bev_mask.eval()
            np.testing.assert_array_equal(img_mask_out, 0.0)
            # Because of the third condition, we expect to be keeping bev
            np.testing.assert_array_equal(bev_mask_out, 1.0)

        ##############################################
        # Test-Case 5 : Kill img, Kill bev, third flip
        ##############################################
        # Let's flip the third chance and keep img instead
        rand_choice = [0.53, 0.83, 0.61]
        rand_choice_tensor = tf.convert_to_tensor(rand_choice)
        p_img = tf.constant(0.0)
        p_bev = tf.constant(0.1)

        img_mask, bev_mask = rpn_model.create_path_drop_masks(
            p_img, p_bev, rand_choice_tensor)

        with self.test_session():
            img_mask_out = img_mask.eval()
            bev_mask_out = bev_mask.eval()
            # Because of the third condition, we expect to be keeping img
            np.testing.assert_array_equal(img_mask_out, 1.0)
            np.testing.assert_array_equal(bev_mask_out, 0.0)

    def test_path_drop_input_multiplication(self):
        # Tests the result of final image/bev inputs
        # based on the path drop decisions

        rpn_model = RpnModel(self.model_config,
                             train_val_test="val",
                             dataset=self.dataset)
        rpn_model.build()
        # Shape of input feature map
        dummy_img_feature_shape = [1, 30, 50, 2]
        random_values = np.random.randint(low=1.0,
                                          high=256.0,
                                          size=2).astype(np.float32)

        dummy_img_feature_map = tf.fill(dummy_img_feature_shape,
                                        random_values[0])
        # Assume both features map are the same size, this is not
        # the case inside the network
        dummy_bev_feature_map = tf.fill(dummy_img_feature_shape,
                                        random_values[1])

        ##################################
        # Test-Case 1 : Keep img, Kill bev
        ##################################
        exp_img_input = np.full(dummy_img_feature_shape, random_values[0])
        exp_bev_input = np.full(dummy_img_feature_shape, 0.0)

        p_img = tf.constant(0.6)
        p_bev = tf.constant(0.4)

        # Set the random numbers for testing purposes
        rand_choice = [0.53, 0.83, 0.05]
        rand_choice_tensor = tf.convert_to_tensor(rand_choice)

        img_mask, bev_mask = rpn_model.create_path_drop_masks(
            p_img, p_bev, rand_choice_tensor)

        final_img_input = tf.multiply(dummy_img_feature_map,
                                      img_mask)

        final_bev_input = tf.multiply(dummy_bev_feature_map,
                                      bev_mask)

        with self.test_session():
            final_img_input_out = final_img_input.eval()
            final_bev_input_out = final_bev_input.eval()
            np.testing.assert_array_equal(final_img_input_out,
                                          exp_img_input)
            np.testing.assert_array_equal(final_bev_input_out,
                                          exp_bev_input)

        ##################################
        # Test-Case 2 : Kill img, Keep bev
        ##################################
        exp_img_input = np.full(dummy_img_feature_shape, 0)
        exp_bev_input = np.full(dummy_img_feature_shape, random_values[1])

        p_img = tf.constant(0.4)
        p_bev = tf.constant(0.9)

        img_mask, bev_mask = rpn_model.create_path_drop_masks(
            p_img, p_bev, rand_choice_tensor)

        final_img_input = tf.multiply(dummy_img_feature_map,
                                      img_mask)

        final_bev_input = tf.multiply(dummy_bev_feature_map,
                                      bev_mask)

        with self.test_session():
            final_img_input_out = final_img_input.eval()
            final_bev_input_out = final_bev_input.eval()
            np.testing.assert_array_equal(final_img_input_out,
                                          exp_img_input)
            np.testing.assert_array_equal(final_bev_input_out,
                                          exp_bev_input)
        """


if __name__ == "__main__":
    tf.test.main()
