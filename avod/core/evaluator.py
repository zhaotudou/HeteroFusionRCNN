"""Common functions for evaluating checkpoints.
"""

import time
import os
import numpy as np
from multiprocessing import Process

import tensorflow as tf

from wavedata.tools.obj_detection import obj_utils

from avod.core import box_3d_encoder
from avod.core import evaluator_utils
from avod.core import summary_utils
from avod.core import trainer_utils
from avod.core import box_util

from avod.core.models.avod_model import AvodModel
from avod.core.models.rpn_model import RpnModel

tf.logging.set_verbosity(tf.logging.INFO)

KEY_SUM_RPN_SEG_LOSS = 'sum_rpn_seg_loss'
KEY_SUM_RPN_BIN_CLS_LOSS = 'sum_rpn_bin_cls_loss'
KEY_SUM_RPN_REG_LOSS = 'sum_rpn_reg_loss'
KEY_SUM_RPN_TOTAL_LOSS = 'sum_rpn_total_loss'
KEY_SUM_RPN_SEG_ACC = 'sum_rpn_seg_accuracy'
KEY_SUM_RPN_RECALL_50 = 'sum_rpn_recall_50'
KEY_SUM_RPN_RECALL_70 = 'sum_rpn_recall_70'
KEY_SUM_RPN_LABEL = 'sum_rpn_label'
KEY_SUM_RPN_PROPOSAL = 'sum_rpn_proposal'
KEY_SUM_RPN_IOU2D = 'sum_rpn_iou2d'
KEY_SUM_RPN_IOU3D = 'sum_rpn_iou3d'
KEY_SUM_RPN_ANGLE_RES = 'sum_rpn_angel_residual'

KEY_SUM_AVOD_CLS_LOSS = 'sum_avod_cls_loss'
KEY_SUM_AVOD_BIN_CLS_LOSS = 'sum_avod_bin_cls_loss'
KEY_SUM_AVOD_REG_LOSS = 'sum_avod_reg_loss'
KEY_SUM_AVOD_TOTAL_LOSS = 'sum_avod_total_loss'
KEY_SUM_AVOD_CLS_ACC = 'sum_avod_cls_accuracy'


class Evaluator:

    def __init__(self,
                 model,
                 dataset_config,
                 eval_config,
                 skip_evaluated_checkpoints=True,
                 eval_wait_interval=30,
                 do_kitti_native_eval=True):
        """Evaluator class for evaluating model's detection output.

        Args:
            model: An instance of DetectionModel
            dataset_config: Dataset protobuf configuration
            eval_config: Evaluation protobuf configuration
            skip_evaluated_checkpoints: (optional) Enables checking evaluation
                results directory and if the folder names with the checkpoint
                index exists, it 'assumes' that checkpoint has already been
                evaluated and skips that checkpoint.
            eval_wait_interval: (optional) The number of seconds between
                looking for a new checkpoint.
            do_kitti_native_eval: (optional) flag to enable running kitti native
                eval code.
        """

        # Get model configurations
        self.model = model
        self.dataset_config = dataset_config
        self.eval_config = eval_config

        self.model_config = model.model_config
        self.model_name = self.model_config.model_name
        self.full_model = isinstance(self.model, AvodModel)

        self.paths_config = self.model_config.paths_config
        self.checkpoint_dir = self.paths_config.checkpoint_dir

        self.skip_evaluated_checkpoints = skip_evaluated_checkpoints
        self.eval_wait_interval = eval_wait_interval

        self.do_kitti_native_eval = do_kitti_native_eval

        # Create a variable tensor to hold the global step
        self.global_step_tensor = tf.Variable(
            0, trainable=False, name='global_step')
        
        self._batch_size = eval_config.batch_size
        eval_mode = eval_config.eval_mode
        if eval_mode not in ['val', 'test']:
            raise ValueError('Evaluation mode can only be set to `val`'
                             'or `test`')

        if not os.path.exists(self.checkpoint_dir):
            raise ValueError('{} must have at least one checkpoint entry.'
                             .format(self.checkpoint_dir))

        if self.do_kitti_native_eval:
            if self.eval_config.eval_mode == 'val':
                # Copy kitti native eval code into the predictions folder
                evaluator_utils.copy_kitti_native_code(
                    self.model_config.checkpoint_name)

        allow_gpu_mem_growth = self.eval_config.allow_gpu_mem_growth
        if allow_gpu_mem_growth:
            # GPU memory config
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = allow_gpu_mem_growth
            self._sess = tf.Session(config=config)
        else:
            self._sess = tf.Session()

        # The model should return a dictionary of predictions
        self._prediction_dict = self.model.build()
        if eval_mode == 'val':
            # Setup loss and summary writer in val mode only
            self._loss_dict, self._total_loss = \
                self.model.loss(self._prediction_dict)

            self.summary_writer, self.summary_merged = \
                evaluator_utils.set_up_summary_writer(self.model_config,
                                                      self._sess)

        else:
            self._loss_dict = None
            self._total_loss = None
            self.summary_writer = None
            self.summary_merged = None

        self._saver = tf.train.Saver()

        # Add maximum memory usage summary op
        # This op can only be run on device with gpu
        # so it's skipped on travis
        is_travis = 'TRAVIS' in os.environ
        if not is_travis:
            # tf 1.4
            # tf.summary.scalar('bytes_in_use',
            #                   tf.contrib.memory_stats.BytesInUse())
            tf.summary.scalar('max_bytes',
                              tf.contrib.memory_stats.MaxBytesInUse())

    def run_checkpoint_once(self, checkpoint_to_restore):
        """Evaluates network metrics once over all the validation samples.

        Args:
            checkpoint_to_restore: The directory of the checkpoint to restore.
        """

        self._saver.restore(self._sess, checkpoint_to_restore)

        data_split = self.dataset_config.data_split
        predictions_base_dir = self.paths_config.pred_dir

        num_samples = self.model.dataset.num_samples
        train_val_test = self.model._train_val_test

        validation = train_val_test == 'val'

        global_step = trainer_utils.get_global_step(
            self._sess, self.global_step_tensor)

        if self.full_model:
            # Add folders to save predictions
            avod_predictions_dir = predictions_base_dir + \
                "/final_predictions_and_scores/{}/{}".format(
                    data_split, global_step)
            trainer_utils.create_dir(avod_predictions_dir)
            
            if validation:
                eval_avod_stats = self._create_avod_stats_dict()
        else:
            # Add folders to save proposals
            prop_score_predictions_dir = predictions_base_dir + \
                "/proposals_and_scores/{}/{}".format(
                    data_split, global_step)
            trainer_utils.create_dir(prop_score_predictions_dir)
            os.system("ln -snf {} {}".format(prop_score_predictions_dir, self.model.dataset.proposal_dir))
            
            if validation:
                eval_rpn_stats = self._create_rpn_stats_dict()
                # Add folders to save proposals info, i.e. IoUs with GT Boxes
                prop_info_dir = predictions_base_dir + \
                    "/proposals_info/{}/{}".format(
                        data_split, global_step)
                trainer_utils.create_dir(prop_info_dir)
                os.system("ln -snf {} {}".format(prop_info_dir, self.model.dataset.proposal_info_dir))

        num_valid_samples = 0

        # Keep track of feed_dict and inference time
        total_feed_dict_time = []
        total_inference_time = []

        # Run through a single epoch
        current_epoch = self.model.dataset.epochs_completed
        while current_epoch == self.model.dataset.epochs_completed:

            # Keep track of feed_dict speed
            start_time = time.time()
            feed_dict = self.model.create_feed_dict(self._batch_size)
            feed_dict_time = time.time() - start_time

            # Get sample name from model
            sample_names = self.model._samples_info

            if self.full_model:
                assert(len(sample_names) == 1)
                avod_file_path = avod_predictions_dir + "/{}.txt".format(sample_names[0])
                if os.path.exists(avod_file_path):
                    continue
            else:
                # File paths for saving proposals and predictions
                rpn_file_paths = [prop_score_predictions_dir + "/{}.txt".format(sample_name) 
                                    for sample_name in sample_names]
                if os.path.exists(rpn_file_paths[0]):
                    continue

            num_valid_samples += self._batch_size
            print("Step {}: {} / {}, Inference on sample {}".format(
                global_step, num_valid_samples, num_samples, ' '.join(sample_names)))

            # Do predictions, loss calculations, and summaries
            if validation:
                if self.summary_merged is not None:
                    predictions, eval_losses, eval_total_loss, summary_out = \
                        self._sess.run([self._prediction_dict,
                                        self._loss_dict,
                                        self._total_loss,
                                        self.summary_merged],
                                       feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_out, global_step)

                else:
                    predictions, eval_losses, eval_total_loss = \
                        self._sess.run([self._prediction_dict,
                                        self._loss_dict,
                                        self._total_loss],
                                       feed_dict=feed_dict)

                if self.full_model:
                    # Save predictions
                    self.save_avod_predicted_boxes_3d_and_scores(predictions, avod_file_path)

                    self._update_avod_box_cls_loc_losses(
                        eval_avod_stats,
                        eval_losses,
                        eval_total_loss,
                        global_step)
                    '''
                    if box_rep != 'box_3d':
                        # Save box corners for all box reps
                        # except for box_3d which is not a corner rep
                        predicted_box_corners_and_scores = \
                            self.get_avod_predicted_box_corners_and_scores(
                                predictions, box_rep)
                        np.savetxt(avod_box_corners_file_path,
                                   predicted_box_corners_and_scores,
                                   fmt='%.5f')
                    '''
                else:
                    rpn_segmentation_loss = eval_losses[RpnModel.LOSS_RPN_SEGMENTATION]
                    rpn_bin_classification_loss = eval_losses[RpnModel.LOSS_RPN_BIN_CLASSIFICATION]
                    rpn_regression_loss = eval_losses[RpnModel.LOSS_RPN_REGRESSION]

                    self._update_rpn_losses(eval_rpn_stats,
                                            rpn_segmentation_loss,
                                            rpn_bin_classification_loss,
                                            rpn_regression_loss,
                                            eval_total_loss,
                                            global_step)

                    # Save proposals
                    self.save_rpn_proposals_and_scores(predictions, rpn_file_paths)

                    # Save proposals info
                    prop_info_files = [prop_info_dir + "/{}.txt".format(sample_name) 
                                        for sample_name in sample_names]
                    self.calculate_proposals_info(rpn_file_paths, 
                                                  sample_names,
                                                  prop_info_files,
                                                  eval_rpn_stats, 
                                                  global_step)

                # Calculate accuracies
                self.get_cls_accuracy(predictions,
                                      eval_avod_stats if self.full_model else eval_rpn_stats,
                                      global_step)
                print("Step {}: Total time {} s".format(
                    global_step, time.time() - start_time))

            else:
                # Test mode --> train_val_test == 'test'
                inference_start_time = time.time()
                # Don't calculate loss or run summaries for test
                predictions = self._sess.run(self._prediction_dict,
                                             feed_dict=feed_dict)
                inference_time = time.time() - inference_start_time

                # Add times to list
                total_feed_dict_time.append(feed_dict_time)
                total_inference_time.append(inference_time)

                if self.full_model:
                    self.save_avod_predicted_boxes_3d_and_scores(predictions, avod_file_path)
                else:
                    self.save_rpn_proposals_and_scores(predictions, rpn_file_paths)

        # end while current_epoch == model.dataset.epochs_completed:

        if validation:
            if self.full_model:
                self.save_prediction_stats(
                    eval_avod_stats,
                    num_valid_samples,
                    global_step,
                    predictions_base_dir)

                # Kitti native evaluation, do this during validation
                # and when running Avod model.
                # Store predictions in kitti format
                if self.do_kitti_native_eval:
                    self.run_kitti_native_eval(global_step)
            else:
                self.save_rpn_stats(eval_rpn_stats,
                                    num_valid_samples,
                                    global_step,
                                    predictions_base_dir)

        else:
            # Test mode --> train_val_test == 'test'
            evaluator_utils.print_inference_time_statistics(
                total_feed_dict_time, total_inference_time)

        print("Step {}: Finished evaluation, results saved to {}".format(
            global_step, prop_score_predictions_dir))

    def run_latest_checkpoints(self):
        """Evaluation function for evaluating all the existing checkpoints.
        This function just runs through all the existing checkpoints.

        Raises:
            ValueError: if model.checkpoint_dir doesn't have at least one
                element.
        """

        if not os.path.exists(self.checkpoint_dir):
            raise ValueError('{} must have at least one checkpoint entry.'
                             .format(self.checkpoint_dir))

        # Load the latest checkpoints available
        trainer_utils.load_checkpoints(self.checkpoint_dir,
                                       self._saver)

        num_checkpoints = len(self._saver.last_checkpoints)

        if self.skip_evaluated_checkpoints:
            already_evaluated_ckpts = self.get_evaluated_ckpts(
                self.model_config, self.model_name)

        ckpt_indices = np.asarray(self.eval_config.ckpt_indices)
        if ckpt_indices is not None:
            if ckpt_indices[0] == -1:
                # Restore the most recent checkpoint
                ckpt_idx = num_checkpoints - 1
                ckpt_indices = [ckpt_idx]
            for ckpt_idx in ckpt_indices:
                checkpoint_to_restore = self._saver.last_checkpoints[ckpt_idx]
                self.run_checkpoint_once(checkpoint_to_restore)

        else:
            last_checkpoint_id = -1
            number_of_evaluations = 0
            # go through all existing checkpoints
            for ckpt_idx in range(num_checkpoints):
                checkpoint_to_restore = self._saver.last_checkpoints[ckpt_idx]
                ckpt_id = evaluator_utils.strip_checkpoint_id(
                    checkpoint_to_restore)

                # Check if checkpoint has been evaluated already
                already_evaluated = ckpt_id in already_evaluated_ckpts
                if already_evaluated or ckpt_id <= last_checkpoint_id:
                    number_of_evaluations = max((ckpt_idx + 1,
                                                 number_of_evaluations))
                    continue

                self.run_checkpoint_once(checkpoint_to_restore)
                number_of_evaluations += 1

                # Save the id of the latest evaluated checkpoint
                last_checkpoint_id = ckpt_id

    def repeated_checkpoint_run(self):
        """Periodically evaluates the checkpoints inside the `checkpoint_dir`.

        This function evaluates all the existing checkpoints as they are being
        generated. If there are none, it sleeps until new checkpoints become
        available. Since there is no synchronization guarantee for the trainer
        and evaluator, at each iteration it reloads all the checkpoints and
        searches for the last checkpoint to continue from. This is meant to be
        called in parallel to the trainer to evaluate the models regularly.

        Raises:
            ValueError: if model.checkpoint_dir doesn't have at least one
                element.
        """

        if not os.path.exists(self.checkpoint_dir):
            raise ValueError('{} must have at least one checkpoint entry.'
                             .format(self.checkpoint_dir))

        if self.skip_evaluated_checkpoints:
            already_evaluated_ckpts = self.get_evaluated_ckpts(
                self.model_config, self.full_model)
        tf.logging.info(
            'Starting evaluation at ' +
            time.strftime(
                '%Y-%m-%d-%H:%M:%S',
                time.gmtime()))

        last_checkpoint_id = -1
        number_of_evaluations = 0
        while True:
            # Load current checkpoints available
            trainer_utils.load_checkpoints(self.checkpoint_dir,
                                           self._saver)
            num_checkpoints = len(self._saver.last_checkpoints)

            start = time.time()

            if number_of_evaluations >= num_checkpoints:
                tf.logging.info('No new checkpoints found in %s.'
                                'Will try again in %d seconds',
                                self.checkpoint_dir,
                                self.eval_wait_interval)
            else:
                for ckpt_idx in range(num_checkpoints):
                    checkpoint_to_restore = \
                        self._saver.last_checkpoints[ckpt_idx]
                    ckpt_id = evaluator_utils.strip_checkpoint_id(
                        checkpoint_to_restore)

                    # Check if checkpoint has been evaluated already
                    already_evaluated = ckpt_id in already_evaluated_ckpts
                    if already_evaluated or ckpt_id <= last_checkpoint_id:
                        number_of_evaluations = max((ckpt_idx + 1,
                                                     number_of_evaluations))
                        continue

                    self.run_checkpoint_once(checkpoint_to_restore)
                    number_of_evaluations += 1

                    # Save the id of the latest evaluated checkpoint
                    last_checkpoint_id = ckpt_id

            time_to_next_eval = start + self.eval_wait_interval - time.time()
            if time_to_next_eval > 0:
                time.sleep(time_to_next_eval)

    def _update_rpn_losses(self,
                           eval_rpn_stats,
                           rpn_segmentation_loss,
                           rpn_bin_classification_loss,
                           rpn_regression_loss,
                           eval_total_loss,
                           global_step):
        """Helper function to calculate the evaluation average losses.

        Args:
            eval_rpn_stats: A dictionary containing all the average
                losses.
            rpn_objectness_loss: A scalar loss of rpn objectness.
            rpn_regression_loss: A scalar loss of rpn objectness.
            eval_total_loss: A scalar loss of rpn total loss.
            global_step: Global step at which the metrics are computed.
        """

        if self.full_model:
            # The full model total_loss will be the sum of Rpn and Avod
            # so calculate the total rpn loss instead
            rpn_total_loss = rpn_segmentation_loss + rpn_bin_classification_loss +  rpn_regression_loss
        else:
            rpn_total_loss = eval_total_loss

        print("Step {}: Eval RPN Loss: segmentation {:.3f}, "
              "bin_cls {:.3f}, regression {:.3f}, total {:.3f}".format(
                    global_step,
                    rpn_segmentation_loss,
                    rpn_bin_classification_loss,
                    rpn_regression_loss,
                    rpn_total_loss))

        # Get the loss sums from the losses dict
        sum_rpn_seg_loss = eval_rpn_stats[KEY_SUM_RPN_SEG_LOSS]
        sum_rpn_bin_cls_loss = eval_rpn_stats[KEY_SUM_RPN_BIN_CLS_LOSS]
        sum_rpn_reg_loss = eval_rpn_stats[KEY_SUM_RPN_REG_LOSS]
        sum_rpn_total_loss = eval_rpn_stats[KEY_SUM_RPN_TOTAL_LOSS]

        sum_rpn_seg_loss += rpn_segmentation_loss
        sum_rpn_bin_cls_loss += rpn_bin_classification_loss
        sum_rpn_reg_loss += rpn_regression_loss
        sum_rpn_total_loss += rpn_total_loss

        # update the losses sums
        eval_rpn_stats.update({KEY_SUM_RPN_SEG_LOSS:
                                sum_rpn_seg_loss})

        eval_rpn_stats.update({KEY_SUM_RPN_BIN_CLS_LOSS:
                                sum_rpn_bin_cls_loss})
        
        eval_rpn_stats.update({KEY_SUM_RPN_REG_LOSS:
                                sum_rpn_reg_loss})

        eval_rpn_stats.update({KEY_SUM_RPN_TOTAL_LOSS:
                                sum_rpn_total_loss})

    def _update_avod_box_cls_loc_losses(self,
                                        eval_avod_stats,
                                        eval_losses,
                                        eval_total_loss,
                                        global_step):
        """Helper function to calculate the evaluation average losses.

        Note: This function evaluates only classification and regression/offsets
            losses.

        Args:
            eval_avod_stats: A dictionary containing all the average
                losses.
            eval_losses: A dictionary containing the current evaluation
                losses.
            eval_total_loss: A scalar loss of model total loss.
            global_step: Global step at which the metrics are computed.
        """

        sum_avod_cls_loss = eval_avod_stats[KEY_SUM_AVOD_CLS_LOSS]
        sum_avod_bin_cls_loss = eval_avod_stats[KEY_SUM_AVOD_BIN_CLS_LOSS]
        sum_avod_reg_loss = eval_avod_stats[KEY_SUM_AVOD_REG_LOSS]
        sum_avod_total_loss = eval_avod_stats[KEY_SUM_AVOD_TOTAL_LOSS]

        # for the full model, we expect a total of 4 losses
        assert (len(eval_losses) > 3)
        avod_classification_loss = \
            eval_losses[AvodModel.LOSS_FINAL_CLASSIFICATION]
        avod_bin_classification_loss = \
            eval_losses[AvodModel.LOSS_FINAL_BIN_CLASSIFICATION]
        avod_regression_loss = \
            eval_losses[AvodModel.LOSS_FINAL_REGRESSION]

        sum_avod_cls_loss += avod_classification_loss
        sum_avod_bin_cls_loss += avod_bin_classification_loss
        sum_avod_reg_loss += avod_regression_loss
        sum_avod_total_loss += eval_total_loss

        eval_avod_stats.update({KEY_SUM_AVOD_CLS_LOSS:
                                 sum_avod_cls_loss})

        eval_avod_stats.update({KEY_SUM_AVOD_BIN_CLS_LOSS:
                                 sum_avod_bin_cls_loss})

        eval_avod_stats.update({KEY_SUM_AVOD_REG_LOSS:
                                 sum_avod_reg_loss})

        eval_avod_stats.update({KEY_SUM_AVOD_TOTAL_LOSS:
                                 sum_avod_total_loss})

        print("Step {}: Eval AVOD Loss: "
              "classification {:.3f}, "
              "bin_classification {:.3f}, "
              "regression {:.3f}, "
              "total {:.3f}".format(
                global_step,
                avod_classification_loss,
                avod_bin_classification_loss,
                avod_regression_loss,
                eval_total_loss))

    def save_rpn_stats(self,
                       eval_rpn_stats,
                       num_valid_samples,
                       global_step,
                       predictions_base_dir):
        """Helper function to save the RPN loss evaluation results.
        """
        sum_rpn_seg_loss = eval_rpn_stats[KEY_SUM_RPN_SEG_LOSS]
        sum_rpn_bin_cls_loss = eval_rpn_stats[KEY_SUM_RPN_BIN_CLS_LOSS]
        sum_rpn_reg_loss = eval_rpn_stats[KEY_SUM_RPN_REG_LOSS]
        sum_rpn_total_loss = eval_rpn_stats[KEY_SUM_RPN_TOTAL_LOSS]
        sum_rpn_seg_accuracy = eval_rpn_stats[KEY_SUM_RPN_SEG_ACC]
        sum_rpn_recall_50 = eval_rpn_stats[KEY_SUM_RPN_RECALL_50]
        sum_rpn_recall_70 = eval_rpn_stats[KEY_SUM_RPN_RECALL_70]
        sum_rpn_label = eval_rpn_stats[KEY_SUM_RPN_LABEL]
        sum_rpn_proposal = eval_rpn_stats[KEY_SUM_RPN_PROPOSAL]
        sum_rpn_iou2d = eval_rpn_stats[KEY_SUM_RPN_IOU2D]
        sum_rpn_iou3d = eval_rpn_stats[KEY_SUM_RPN_IOU3D]
        sum_rpn_angel_res = eval_rpn_stats[KEY_SUM_RPN_ANGLE_RES]

        # Calculate average loss and accuracy
        avg_rpn_seg_loss = sum_rpn_seg_loss / num_valid_samples
        avg_rpn_bin_cls_loss = sum_rpn_bin_cls_loss / num_valid_samples
        avg_rpn_reg_loss = sum_rpn_reg_loss / num_valid_samples
        avg_rpn_total_loss = sum_rpn_total_loss / num_valid_samples
        avg_rpn_seg_accuracy = sum_rpn_seg_accuracy / num_valid_samples

        avg_rpn_proposal = sum_rpn_proposal / num_valid_samples
        total_rpn_recall_50 = sum_rpn_recall_50 / sum_rpn_label
        total_rpn_recall_70 = sum_rpn_recall_70 / sum_rpn_label
        
        avg_rpn_iou2d = sum_rpn_iou2d / sum_rpn_proposal
        avg_rpn_iou3d = sum_rpn_iou3d / sum_rpn_proposal
        avg_rpn_angel_res = sum_rpn_angel_res / sum_rpn_proposal
        print("Step {}: Average RPN Losses: segmentation {:.3f}, "
              "bin_cls {:.3f}, regression {:.3f}, total {:.3f}".format(global_step,
                                                       avg_rpn_seg_loss,
                                                       avg_rpn_bin_cls_loss,
                                                       avg_rpn_reg_loss,
                                                       avg_rpn_total_loss))
        print("Step {}: Average Segmentation Accuracy:{:.3f} ".format(
            global_step,
            avg_rpn_seg_accuracy))
        print("Step {}: Total RPN Recall@3DIoU=0.5: {:.3f}  Recall@3DIoU=0.7: {:.3f}, Average Proposals: {:.3f}".format(
            global_step,
            total_rpn_recall_50, total_rpn_recall_70, avg_rpn_proposal))
        print("Step {}: Average RPN IoU_2D: {:.3f} , IoU_3D: {:.3f}, Average Angel Residual: {:.3f}".format(
            global_step,
            avg_rpn_iou2d, avg_rpn_iou3d, avg_rpn_angel_res))
        
        # Append to end of file
        avg_loss_file_path = predictions_base_dir + '/rpn_avg_losses.csv'
        with open(avg_loss_file_path, 'ba') as fp:
            np.savetxt(fp,
                       np.reshape([global_step,
                                   avg_rpn_seg_loss,
                                   avg_rpn_bin_cls_loss,
                                   avg_rpn_reg_loss,
                                   avg_rpn_total_loss],
                                  (1, 5)),
                       fmt='%d, %.5f, %.5f, %.5f, %5f')

        avg_acc_file_path = predictions_base_dir + '/rpn_avg_seg_acc.csv'
        with open(avg_acc_file_path, 'ba') as fp:
            np.savetxt(
                fp, np.reshape(
                    [global_step, avg_rpn_seg_accuracy],
                    (1, 2)),
                fmt='%d, %.5f')
        
        total_recall_file_path = predictions_base_dir + '/rpn_total_recall.csv'
        with open(total_recall_file_path, 'ba') as fp:
            np.savetxt(
                fp, np.reshape(
                    [global_step, total_rpn_recall_50, total_rpn_recall_70, avg_rpn_proposal, 
                     avg_rpn_iou2d, avg_rpn_iou3d, avg_rpn_angel_res],
                    (1, 7)),
                fmt='%d, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f')

    def save_prediction_stats(self,
                              eval_avod_stats,
                              num_valid_samples,
                              global_step,
                              predictions_base_dir):
        """Helper function to save the AVOD loss evaluation results.

        Args:
            eval_avod_stats: A dictionary containing the loss sums
            num_valid_samples: An int, number of valid evaluated samples
                i.e. samples with valid ground-truth.
            global_step: Global step at which the metrics are computed.
            predictions_base_dir: Base directory for storing the results.
            box_rep: A string, the format of the 3D bounding box
                one of 'box_3d', 'box_8c' etc.
        """
        sum_avod_cls_loss = eval_avod_stats[KEY_SUM_AVOD_CLS_LOSS]
        sum_avod_bin_cls_loss = eval_avod_stats[KEY_SUM_AVOD_BIN_CLS_LOSS]
        sum_avod_reg_loss = eval_avod_stats[KEY_SUM_AVOD_REG_LOSS]
        sum_avod_total_loss = eval_avod_stats[KEY_SUM_AVOD_TOTAL_LOSS]

        sum_avod_cls_accuracy = eval_avod_stats[KEY_SUM_AVOD_CLS_ACC]

        avg_avod_cls_loss = sum_avod_cls_loss / num_valid_samples
        avg_avod_bin_cls_loss = sum_avod_bin_cls_loss / num_valid_samples
        avg_avod_reg_loss = sum_avod_reg_loss / num_valid_samples
        avg_avod_total_loss = sum_avod_total_loss / num_valid_samples

        avg_avod_cls_accuracy = sum_avod_cls_accuracy / num_valid_samples

        print("Step {}: Average AVOD Losses: "
              "cls {:.5f}, "
              "bin_cls {:.5f}, "
              "reg {:.5f}, "
              "total {:.5f} ".format(
                global_step,
                avg_avod_cls_loss,
                avg_avod_bin_cls_loss,
                avg_avod_reg_loss,
                avg_avod_total_loss,
                  ))

        print("Step {}: Average Classification Accuracy: {:.3f} ".format(
            global_step, avg_avod_cls_accuracy))

        # Append to end of file
        avg_loss_file_path = predictions_base_dir + '/avod_avg_losses.csv'
        with open(avg_loss_file_path, 'ba') as fp:
            np.savetxt(fp,
                       [np.hstack(
                        [global_step,
                            avg_avod_cls_loss,
                            avg_avod_bin_cls_loss,
                            avg_avod_reg_loss,
                            avg_avod_total_loss,
                         ]
                        )],
                       fmt='%d, %.5f, %.5f, %.5f, %.5f')

        avg_acc_file_path = predictions_base_dir + '/avod_avg_cls_acc.csv'
        with open(avg_acc_file_path, 'ba') as fp:
            np.savetxt(
                fp, np.reshape(
                    [global_step, avg_avod_cls_accuracy],
                    (1, 2)),
                fmt='%d, %.5f')

    def _create_avod_stats_dict(self):
        """Returns a dictionary of the losses sum for averaging.
        """
        eval_avod_stats = dict()
        # Initialize Avod average losses
        eval_avod_stats[KEY_SUM_AVOD_CLS_LOSS] = 0
        eval_avod_stats[KEY_SUM_AVOD_BIN_CLS_LOSS] = 0
        eval_avod_stats[KEY_SUM_AVOD_REG_LOSS] = 0
        eval_avod_stats[KEY_SUM_AVOD_TOTAL_LOSS] = 0

        eval_avod_stats[KEY_SUM_AVOD_CLS_ACC] = 0

        return eval_avod_stats

    def _create_rpn_stats_dict(self):
        """Returns a dictionary of the losses sum for averaging.
        """
        eval_rpn_stats = dict()

        # Initialize Rpn average losses
        eval_rpn_stats[KEY_SUM_RPN_SEG_LOSS] = 0
        eval_rpn_stats[KEY_SUM_RPN_BIN_CLS_LOSS] = 0
        eval_rpn_stats[KEY_SUM_RPN_REG_LOSS] = 0
        eval_rpn_stats[KEY_SUM_RPN_TOTAL_LOSS] = 0
        eval_rpn_stats[KEY_SUM_RPN_SEG_ACC] = 0

        eval_rpn_stats[KEY_SUM_RPN_RECALL_50] = 0
        eval_rpn_stats[KEY_SUM_RPN_RECALL_70] = 0
        eval_rpn_stats[KEY_SUM_RPN_LABEL] = 0
        eval_rpn_stats[KEY_SUM_RPN_PROPOSAL] = 0
        eval_rpn_stats[KEY_SUM_RPN_IOU2D] = 0
        eval_rpn_stats[KEY_SUM_RPN_IOU3D] = 0
        eval_rpn_stats[KEY_SUM_RPN_ANGLE_RES] = 0
        

        return eval_rpn_stats

    def get_evaluated_ckpts(self,
                            model_config,
                            model_name):
        """Finds the evaluated checkpoints.

        Examines the evaluation average losses file to find the already
        evaluated checkpoints.

        Args:
            model_config: Model protobuf configuration
            model_name: A string representing the model name.

        Returns:
            already_evaluated_ckpts: A list of checkpoint indices, or an
                empty list if no evaluated indices are found.
        """

        already_evaluated_ckpts = []

        # check for previously evaluated checkpoints
        # regardless of model, we are always evaluating rpn, but we do
        # this check based on model in case the evaluator got interrupted
        # and only saved results for one model
        paths_config = model_config.paths_config

        predictions_base_dir = paths_config.pred_dir
        if model_name == 'avod_model':
            avg_loss_file_path = predictions_base_dir + '/avod_avg_losses.csv'
        else:
            avg_loss_file_path = predictions_base_dir + '/rpn_avg_losses.csv'

        if os.path.exists(avg_loss_file_path):
            avg_losses = np.loadtxt(avg_loss_file_path, delimiter=',')
            if avg_losses.ndim == 1:
                # one entry
                already_evaluated_ckpts = np.asarray(
                    [avg_losses[0]], np.int32)
            else:
                already_evaluated_ckpts = np.asarray(avg_losses[:, 0],
                                                     np.int32)

        return already_evaluated_ckpts

    def get_cls_accuracy(self,
                         predictions,
                         eval_stats,
                         global_step):
        """Updates the calculated accuracies for rpn and avod losses.

        Args:
            predictions: A dictionary containing the model outputs.
            eval_avod_stats: A dictionary containing all the avod averaged
                losses.
            eval_rpn_stats: A dictionary containing all the rpn averaged
                losses.
            global_step: Current global step that is being evaluated.
        """

        if self.full_model:
            classification_pred = \
                predictions[AvodModel.PRED_MB_CLASSIFICATION_LOGITS]
            classification_gt = \
                predictions[AvodModel.PRED_MB_CLASSIFICATIONS_GT]
            classification_accuracy = self.calculate_cls_accuracy(
                classification_pred, classification_gt)

            sum_avod_cls_accuracy = eval_stats[KEY_SUM_AVOD_CLS_ACC]
            sum_avod_cls_accuracy += classification_accuracy
            eval_stats.update({KEY_SUM_AVOD_CLS_ACC: sum_avod_cls_accuracy})

            print("Step {}: AVOD Classification Accuracy: {:.3f}".format(
                global_step, classification_accuracy))
        else:
            seg_softmax = predictions[RpnModel.PRED_SEG_SOFTMAX]
            seg_gt = predictions[RpnModel.PRED_SEG_GT]
            segmentation_accuracy = self.calculate_cls_accuracy(seg_softmax, seg_gt)

            # get this from the key
            sum_rpn_seg_accuracy = eval_stats[KEY_SUM_RPN_SEG_ACC]
            sum_rpn_seg_accuracy += segmentation_accuracy
            eval_stats.update({KEY_SUM_RPN_SEG_ACC: sum_rpn_seg_accuracy})
            print("Step {}: RPN Segmentation Accuracy: {:.3f}".format(
                global_step, segmentation_accuracy))


    def calculate_cls_accuracy(self, cls_pred, cls_gt):
        """Calculates accuracy of predicted objectness/classification wrt to
        the labels

        Args:
            cls_pred: A numpy array containing the predicted
            objectness/classification values in the form (mini_batches, 2)
            cls_gt: A numpy array containing the ground truth
            objectness/classification values in the form (mini_batches, 2)

        Returns:
            accuracy: A scalar value representing the accuracy
        """
        correct_prediction = np.equal(np.argmax(cls_pred, -1),
                                      np.argmax(cls_gt, -1))
        accuracy = np.mean(correct_prediction)
        return accuracy

    def save_rpn_proposals_and_scores(self, predictions, rpn_file_paths):
        """Returns the proposals and scores stacked for saving to file.

        Args:
            predictions: A dictionary containing the model outputs.

        Returns:
            proposals_and_scores: A numpy array of shape (number_of_proposals,
                8), containing the rpn proposal boxes and scores.
        """
        nms_indices = predictions[RpnModel.PRED_NMS_INDICES]
        proposals = predictions[RpnModel.PRED_PROPOSALS]
        softmax_scores = predictions[RpnModel.PRED_OBJECTNESS_SOFTMAX]
        
        Batch = nms_indices.shape[0]
        NMS = nms_indices.shape[1]
        for b in range(Batch):
            top_proposals = []
            top_scores = []
            for n in range(NMS):
                idx = nms_indices[b,n]
                if idx == -1:
                    break
                top_proposals.append(proposals[b,idx])
                top_scores.append(softmax_scores[b,idx])

            proposals_and_scores = np.column_stack(
                [np.asarray(top_proposals).reshape((-1,7)), 
                 np.asarray(top_scores).reshape((-1,1))])
            np.savetxt(rpn_file_paths[b], proposals_and_scores, fmt='%.3f')

    def calculate_proposals_info(self, rpn_file_paths, sample_names, prop_info_files, eval_rpn_stats, global_step):
        assert (len(rpn_file_paths) == len(sample_names))
        
        for i in range(len(rpn_file_paths)):
            rpn_file = rpn_file_paths[i]
            sample_name = sample_names[i]
            prop_info_file = prop_info_files[i]

            top_proposals = np.loadtxt(rpn_file).reshape((-1,8))[:,0:7]
            
            obj_labels = obj_utils.read_labels(self.model.dataset.label_dir, int(sample_name))
            # Only use objects that match dataset classes
            obj_labels = self.model.dataset.kitti_utils.filter_labels(obj_labels)

            label_boxes_3d = np.asarray(
                [box_3d_encoder.object_label_to_box_3d(obj_label)
                 for obj_label in obj_labels])
            label_classes = [
                self.model.dataset.kitti_utils.class_str_to_index(obj_label.type)
                for obj_label in obj_labels]
            label_classes = np.asarray(label_classes, dtype=np.int32)

            recall_50, recall_70, iou2ds, iou3ds, iou3ds_gt_boxes, iou3ds_gt_cls = \
                box_util.compute_recall_iou(top_proposals, label_boxes_3d, label_classes)
            
            num_props = top_proposals.shape[0]
            num_labels = label_boxes_3d.shape[0]

            proposals_info = np.column_stack(
                [iou2ds.reshape((num_props,1)),
                 iou3ds.reshape((num_props,1)),
                 iou3ds_gt_boxes.reshape((num_props,7)),
                 iou3ds_gt_cls.reshape((num_props,1))])
            np.savetxt(prop_info_file, proposals_info, fmt='%.3f')
    
            sum_rpn_recall_50 = eval_rpn_stats[KEY_SUM_RPN_RECALL_50]
            sum_rpn_recall_70 = eval_rpn_stats[KEY_SUM_RPN_RECALL_70]
            sum_rpn_label = eval_rpn_stats[KEY_SUM_RPN_LABEL]
            sum_rpn_proposal = eval_rpn_stats[KEY_SUM_RPN_PROPOSAL]
            sum_rpn_iou2d = eval_rpn_stats[KEY_SUM_RPN_IOU2D]
            sum_rpn_iou3d = eval_rpn_stats[KEY_SUM_RPN_IOU3D]
            sum_rpn_angle_res = eval_rpn_stats[KEY_SUM_RPN_ANGLE_RES]
            sum_rpn_recall_50 += recall_50
            sum_rpn_recall_70 += recall_70
            sum_rpn_label += num_labels
            sum_rpn_proposal += num_props
            sum_rpn_iou2d += np.sum(iou2ds)
            sum_rpn_iou3d += np.sum(iou3ds)
            sum_rpn_angle_res += np.sum(np.absolute(top_proposals[:,6] - iou3ds_gt_boxes[:,6]))
            eval_rpn_stats.update({KEY_SUM_RPN_RECALL_50: sum_rpn_recall_50})
            eval_rpn_stats.update({KEY_SUM_RPN_RECALL_70: sum_rpn_recall_70})
            eval_rpn_stats.update({KEY_SUM_RPN_LABEL: sum_rpn_label})
            eval_rpn_stats.update({KEY_SUM_RPN_PROPOSAL: sum_rpn_proposal})
            eval_rpn_stats.update({KEY_SUM_RPN_IOU2D: sum_rpn_iou2d})
            eval_rpn_stats.update({KEY_SUM_RPN_IOU3D: sum_rpn_iou3d})
            eval_rpn_stats.update({KEY_SUM_RPN_ANGLE_RES: sum_rpn_angle_res})
            print("Step {}: RPN Recall@3DIoU=0.5: {:.3f}  Recall@3DIoU=0.7: {:.3f}, num proposals: {:.3f}".format(
                global_step, recall_50 / num_labels, recall_70 / num_labels, num_props))

    def save_avod_predicted_boxes_3d_and_scores(self, predictions, avod_file_path):
        """Returns the predictions and scores stacked for saving to file.

        Args:
            predictions: A dictionary containing the model outputs.
            box_rep: A string indicating the format of the 3D bounding
                boxes i.e. 'box_3d', 'box_8c' etc.

        Returns:
            predictions_and_scores: A numpy array of shape
                (number_of_predicted_boxes, 9), containing the final prediction
                boxes, orientations, scores, and types.
        """

        final_pred_boxes_3d = predictions[AvodModel.PRED_TOP_PREDICTION_BOXES_3D]

        # Append score and class index (object type)
        final_pred_softmax = predictions[AvodModel.PRED_TOP_PREDICTION_SOFTMAX]

        # Find max class score index
        not_bkg_scores = final_pred_softmax[:, 1:]
        final_pred_types = np.argmax(not_bkg_scores, axis=1)

        # Take max class score (ignoring background)
        final_pred_scores = np.array([])
        for pred_idx in range(len(final_pred_boxes_3d)):
            all_class_scores = not_bkg_scores[pred_idx]
            max_class_score = all_class_scores[final_pred_types[pred_idx]]
            final_pred_scores = np.append(final_pred_scores, max_class_score)

        # Stack into prediction format
        predictions_and_scores = np.column_stack(
            [final_pred_boxes_3d,
             final_pred_scores,
             final_pred_types])

        np.savetxt(avod_file_path, predictions_and_scores, fmt='%.5f')

    def get_avod_predicted_box_corners_and_scores(self,
                                                  predictions,
                                                  box_rep):

        if box_rep in ['box_8c', 'box_8co']:
            final_pred_box_corners = predictions[AvodModel.PRED_TOP_BOXES_8C]
        elif box_rep in ['box_4c', 'box_4ca']:
            final_pred_box_corners = predictions[AvodModel.PRED_TOP_BOXES_4C]

        # Append score and class index (object type)
        final_pred_softmax = predictions[
            AvodModel.PRED_TOP_CLASSIFICATION_SOFTMAX]

        # Find max class score index
        not_bkg_scores = final_pred_softmax[:, 1:]
        final_pred_types = np.argmax(not_bkg_scores, axis=1)

        # Take max class score (ignoring background)
        final_pred_scores = np.array([])
        for pred_idx in range(len(final_pred_box_corners)):
            all_class_scores = not_bkg_scores[pred_idx]
            max_class_score = all_class_scores[final_pred_types[pred_idx]]
            final_pred_scores = np.append(final_pred_scores, max_class_score)

        if box_rep in ['box_8c', 'box_8co']:
            final_pred_box_corners = np.reshape(final_pred_box_corners,
                                                [-1, 24])
        # Stack into prediction format
        predictions_and_scores = np.column_stack(
            [final_pred_box_corners,
             final_pred_scores,
             final_pred_types])

        return predictions_and_scores

    def run_kitti_native_eval(self, global_step):
        """Calls the kitti native C++ evaluation code.

        It first saves the predictions in kitti format. It then creates two
        child processes to run the evaluation code. The native evaluation
        hard-codes the IoU threshold inside the code, so hence its called
        twice for each IoU separately.

        Args:
            global_step: Global step of the current checkpoint to be evaluated.
        """

        # Kitti native evaluation, do this during validation
        # and when running Avod model.
        # Store predictions in kitti format
        evaluator_utils.save_predictions_in_kitti_format(
            self.model,
            self.model_config.checkpoint_name,
            self.dataset_config.data_split,
            self.eval_config.kitti_score_threshold,
            global_step)

        checkpoint_name = self.model_config.checkpoint_name
        kitti_score_threshold = self.eval_config.kitti_score_threshold

        # Create a separate processes to run the native evaluation
        native_eval_proc = Process(
            target=evaluator_utils.run_kitti_native_script, args=(
                checkpoint_name, kitti_score_threshold, global_step))
        native_eval_proc_05_iou = Process(
            target=evaluator_utils.run_kitti_native_script_with_05_iou,
            args=(checkpoint_name, kitti_score_threshold, global_step))
        # Don't call join on this cuz we do not want to block
        # this will cause one zombie process - should be fixed later.
        native_eval_proc.start()
        native_eval_proc_05_iou.start()
