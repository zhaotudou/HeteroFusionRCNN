import os
import argparse

import numpy as np
import tensorflow as tf
import horovod.tensorflow
from tensorflow.python.framework import graph_util
from tensorflow.python.ops import variable_scope

import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core.models.avod_model import AvodModel
from avod.core.models.rpn_model import RpnModel
from avod.core.evaluator import Evaluator

dir(tf.contrib)
np.set_printoptions(formatter={"float": lambda x: "{0:0.5f}".format(x)})

SAVE_INFERENCE_MODEL = True
OUTPUT_FINAL_PREDICTION = True
TEST_PB_GRAPH = True
RPN_CONFIG = "../../avod/configs/rpn_multiclass.config"
RCNN_CONFIG = "../../avod/configs/avod_multiclass_rpn60k.config"
DATA_SPLIT = "val"
EVAL_MODE = "test"
SAVE_RPN_FEATURE = True

# np.random.seed(3)


def get_checkpoint_filepath(checkpoint_dir, checkpoint_idx=-1):
    # print("checkpoint dir: ", checkpoint_dir)
    all_checkpoint_states = tf.train.get_checkpoint_state(checkpoint_dir)
    all_checkpoint_paths = all_checkpoint_states.all_model_checkpoint_paths
    assert len(all_checkpoint_paths) > 0, (
        "No checkpoints in directory " + checkpoint_dir
    )
    checkpoint_path = all_checkpoint_paths[checkpoint_idx]
    return checkpoint_path


def get_inference_model_meta(graph, model, checkpoint_idx):
    checkpoint_dir = model.model_config.paths_config.checkpoint_dir
    checkpoint_path = get_checkpoint_filepath(checkpoint_dir, checkpoint_idx)
    inference_model_path = os.path.join(checkpoint_path, "inference_model")
    if SAVE_INFERENCE_MODEL:
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=graph, config=config)
        with sess:
            saver.restore(sess, checkpoint_path)
            saver.save(sess, save_path=inference_model_path)
            print("inference model saved to: ", inference_model_path)
    meta_path = inference_model_path + ".meta"
    # print("meta filepath: ", meta_path)
    return checkpoint_path, meta_path


def get_all_tensors():
    graph = tf.get_default_graph()
    tensors_per_node = [node.values() for node in graph.get_operations()]
    tensor_names = [tensor.name for tensors in tensors_per_node for tensor in tensors]
    return tensor_names


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        getter = lambda next_creator, **kwargs: variable_creator(**kwargs)
        with variable_scope.variable_creator_scope(getter):
            tf.import_graph_def(graph_def, name="pointrcnn")
    return graph


def variable_creator(**kwargs):
    kwargs["use_resource"] = False
    return variable_scope.default_variable_creator(None, **kwargs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rpn_config",
        nargs="?",
        type=str,
        default=RPN_CONFIG,
        help="Path to the RPN pipeline config",
    )

    parser.add_argument(
        "--rcnn_config",
        nargs="?",
        type=str,
        default=RCNN_CONFIG,
        help="Path to the RCNN pipeline config",
    )

    args = parser.parse_args()

    RPN_CONFIG = args.rpn_config
    RCNN_CONFIG = args.rcnn_config

    # RPN Model
    rpn_model_config, _, rpn_eval_config, rpn_dataset_config = config_builder.get_configs_from_pipeline_file(
        RPN_CONFIG, is_training=False
    )
    rpn_eval_config.eval_mode = EVAL_MODE
    rpn_eval_config.save_rpn_feature = SAVE_RPN_FEATURE
    rpn_dataset_config.data_split = DATA_SPLIT

    # Convert to object to overwrite repeated fields
    rpn_dataset_config = config_builder.proto_to_obj(rpn_dataset_config)

    # Remove augmentation during evaluation
    rpn_dataset_config.aug_list = []

    # rpn_dataset_config.aug_list = []
    rpn_dataset = DatasetBuilder.build_kitti_dataset(
        rpn_dataset_config, use_defaults=False
    )
    with tf.Graph().as_default() as rpn_graph:
        getter = lambda next_creator, **kwargs: variable_creator(**kwargs)
        with variable_scope.variable_creator_scope(getter):
            rpn_model = RpnModel(
                rpn_model_config,
                train_val_test=rpn_eval_config.eval_mode,
                dataset=rpn_dataset,
                batch_size=rpn_eval_config.batch_size,
            )
            rpn_model.build()
            rpn_checkpoint_path, rpn_meta_filepath = get_inference_model_meta(
                rpn_graph, rpn_model, 30
            )

    # RCNN Model
    rcnn_model_config, _, rcnn_eval_config, rcnn_dataset_config = config_builder.get_configs_from_pipeline_file(
        RCNN_CONFIG, is_training=False
    )
    rcnn_eval_config.eval_mode = EVAL_MODE
    # rcnn_dataset_config.aug_list = []
    rcnn_dataset = DatasetBuilder.build_kitti_dataset(
        rcnn_dataset_config, use_defaults=False
    )

    with tf.Graph().as_default() as rcnn_graph:
        getter = lambda next_creator, **kwargs: variable_creator(**kwargs)
        with variable_scope.variable_creator_scope(getter):
            rcnn_model = AvodModel(
                rcnn_model_config,
                train_val_test=rcnn_eval_config.eval_mode,
                dataset=rcnn_dataset,
                batch_size=rcnn_eval_config.batch_size,
            )
            rcnn_model.build()
            rcnn_checkpoint_path, rcnn_meta_filepath = get_inference_model_meta(
                rcnn_graph, rcnn_model, -1
            )

    graph = tf.Graph()
    with graph.as_default():
        # load RPN graph
        rpn_saver = tf.train.import_meta_graph(
            rpn_meta_filepath, import_scope="rpn", clear_devices=True
        )
        input_tensor_names = [
            "rpn/pc_input/pc_inputs_pl:0",
            "rpn/img_input/img_input_pl:0",
            "rpn/sample_info/frame_calib_p2:0",
        ]
        input_tensors = [
            graph.get_tensor_by_name(tensor_name) for tensor_name in input_tensor_names
        ]

        pl_names = ["point_cloud", "image_input", "stereo_calib_p2"]

        # rpn_tensors = get_all_tensors()
        # print([tensor for tensor in rpn_tensors if "rpn/output" in tensor])

        rpn_to_rcnn = dict()
        rpn_to_rcnn["rpn/output_pts:0"] = "pl_rpn_feature/rpn_pts_pl:0"
        rpn_to_rcnn["rpn/output_fts:0"] = "pl_rpn_feature/rpn_fts_pl:0"
        rpn_to_rcnn["rpn/output_foreground_mask:0"] = "pl_rpn_feature/rpn_fg_mask_pl:0"
        rpn_to_rcnn["rpn/output_intensities:0"] = "pl_rpn_feature/rpn_intensity_pl:0"
        rpn_to_rcnn["rpn/output_proposals:0"] = "pl_proposals/proposals_pl:0"
        rpn_to_rcnn["rpn/img_input/img_input_pl:0"] = "img_input/img_input_pl:0"
        rpn_to_rcnn["rpn/sample_info/frame_calib_p2:0"] = "sample_info/frame_calib_p2:0"

        # connect RPN and RCNN
        input_map = dict()
        for key, value in rpn_to_rcnn.items():
            tensor = graph.get_tensor_by_name(key)
            input_map[value] = tensor

        # load RCNN graph
        rcnn_saver = tf.train.import_meta_graph(
            rcnn_meta_filepath,
            import_scope="rcnn",
            input_map=input_map,
            clear_devices=True,
        )

        if OUTPUT_FINAL_PREDICTION:
            output_tensor_names = [
                "rcnn/final_boxes:0",
                "rcnn/final_box_classes:0",
                "rcnn/final_box_class_scores:0",
                # "rcnn/before_final_softmax:0",
                # "rpn/output_proposal_objectness:0",
                # "rpn/output_proposals:0"
            ]
        else:
            output_tensor_names = [
                "rcnn/output_reg_boxes_3d:0",
                "rcnn/output_cls_softmax:0",
                "rcnn/output_non_empty_box_mask:0",
                "rcnn/output_nms_indices:0",
                "rcnn/output_num_boxes_before_padding:0",
            ]

        output_tensors = [
            graph.get_tensor_by_name(tensor_name) for tensor_name in output_tensor_names
        ]

    # create feed dict - test data
    samples = rpn_dataset.load_samples(
        [0], model="rpn", pc_sample_pts=rpn_model._pc_sample_pts
    )
    # samples[0]['image_input'] = np.random.randn(720,1080,3)
    print(
        "Use point cloud of Sample {} as test data ".format(
            rpn_dataset.sample_list[0].name
        )
    )
    feed_dict = dict()
    for input_tensor, pl_name in zip(input_tensors, pl_names):
        feed_dict[input_tensor] = [samples[0][pl_name]]

    # try inference
    input_graph_def = graph.as_graph_def()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=config)
    with sess:
        # load varibales
        rpn_saver.restore(sess, rpn_checkpoint_path)
        rcnn_saver.restore(sess, rcnn_checkpoint_path)

        outputs = sess.run(output_tensors, feed_dict=feed_dict)

        # for node in sess.graph_def.node:
        #         if "rcnn/output" in node.name:
        #             print(node.name)

        # freeze graph
        graph_name = "{}_{}.pb".format(
            rpn_checkpoint_path.split("/")[-1], rcnn_checkpoint_path.split("/")[-1]
        )
        output_graph_path = os.path.join(rcnn_checkpoint_path, graph_name)
        output_node_names = [
            tensor_name.split(":")[0] for tensor_name in output_tensor_names
        ]
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, input_graph_def, output_node_names
        )
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        print("saved final graph to: ", output_graph_path)

    if OUTPUT_FINAL_PREDICTION:

        final_boxes = outputs[0]
        final_box_classes = outputs[1]
        final_box_class_scores = outputs[2]
        # before_final_softmax = outputs[3]
        # proposal_objectness = outputs[4]
        # rpn_proposals = outputs[5]

        final_boxes, unique_idxes = np.unique(final_boxes, axis=0, return_index=True)
        final_box_classes = np.expand_dims(final_box_classes[unique_idxes], axis=1)
        final_box_class_scores = np.expand_dims(
            final_box_class_scores[unique_idxes], axis=1
        )

        # print("final boxes shape: ", final_boxes.shape)
        # print("final box class scores shape: ", final_box_class_scores.shape)
        # print("final box classes shape: ", final_box_classes.shape)
        final_prediction = np.hstack(
            (final_boxes, final_box_class_scores, final_box_classes)
        )
        print(
            "Detected following objects in sample {}:".format(
                rpn_dataset.sample_list[0].name
            )
        )
        print(final_prediction)
        if TEST_PB_GRAPH:
            pb_graph = load_graph(output_graph_path)
            input_tensor_name = "pointrcnn/rpn/pc_input/pc_inputs_pl:0"
            input_tensor = pb_graph.get_tensor_by_name(input_tensor_name)
            output_tensor_names = [
                "pointrcnn/rcnn/final_boxes:0",
                "pointrcnn/rcnn/final_box_classes:0",
                "pointrcnn/rcnn/final_box_class_scores:0",
                # "pointrcnn/rcnn/before_final_softmax:0",
                # "pointrcnn/rpn/output_proposal_objectness:0",
                # "pointrcnn/rpn/output_proposals:0"
            ]

            output_tensors = [
                pb_graph.get_tensor_by_name(tensor_name)
                for tensor_name in output_tensor_names
            ]

            with tf.Session(graph=pb_graph) as sess:
                final_boxes, final_box_classes, final_box_class_scores = sess.run(
                    output_tensors,
                    feed_dict={input_tensor: [samples[0]["point_cloud"]]},
                )

            final_boxes, unique_idxes = np.unique(
                final_boxes, axis=0, return_index=True
            )
            final_box_classes = np.expand_dims(final_box_classes[unique_idxes], axis=1)
            final_box_class_scores = np.expand_dims(
                final_box_class_scores[unique_idxes], axis=1
            )

            final_prediction_pb = np.hstack(
                (final_boxes, final_box_class_scores, final_box_classes)
            )
            print(
                "Detected following objects in sample {}:".format(
                    rpn_dataset.sample_list[0].name
                )
            )
            print(final_prediction)
            # print(before_final_softmax)

            print(
                "[PB_GRAPH] Detected following objects in sample {}:".format(
                    rpn_dataset.sample_list[0].name
                )
            )
            print(final_prediction_pb)
    else:
        prediction_dict = dict()
        prediction_dict[AvodModel.PRED_BOXES] = outputs[0]
        prediction_dict[AvodModel.PRED_SOFTMAX] = outputs[1]
        prediction_dict[AvodModel.PRED_NON_EMPTY_BOX_MASK] = outputs[2]
        prediction_dict[AvodModel.PRED_NMS_INDICES] = outputs[3]
        prediction_dict[AvodModel.PRED_NUM_BOXES_BEFORE_PADDING] = outputs[4]

        import pickle

        with open("/tmp/prediction.pickle", "wb") as handle:
            pickle.dump(prediction_dict, handle)

        prediction_save_path = ["/tmp/prediction.txt"]
        Evaluator.save_avod_predicted_boxes_3d_and_scores(
            prediction_dict, prediction_save_path
        )

        print(
            "Detected following objects in sample {}:".format(
                rpn_dataset.sample_list[0].name
            )
        )
        os.system("cat {}".format(prediction_save_path[0]))

    print("%d ops in RCNN graph." % len(output_graph_def.node))
    print("RCNN graph saved to ", output_graph_path)
