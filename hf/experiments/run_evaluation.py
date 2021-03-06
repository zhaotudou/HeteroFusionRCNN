"""Detection model evaluator.

This runs the DetectionModel evaluator.
"""

import argparse
import os

import tensorflow as tf

import hf
import hf.builders.config_builder_util as config_builder
from hf.builders.dataset_builder import DatasetBuilder
from hf.core.models.rcnn_model import RcnnModel
from hf.core.models.rpn_model import RpnModel
from hf.core.evaluator import Evaluator


def evaluate(model_config, eval_config, dataset_config):

    # Parse eval config
    eval_mode = eval_config.eval_mode
    if eval_mode not in ["val", "test"]:
        raise ValueError("Evaluation mode can only be set to `val` or `test`")
    evaluate_repeatedly = eval_config.evaluate_repeatedly
    kitti_native_eval_only = eval_config.kitti_native_eval_only
    kitti_native_eval_step = eval_config.kitti_native_eval_step

    # Parse dataset config
    data_split = dataset_config.data_split
    if data_split == "train":
        dataset_config.data_split_dir = "training"
        dataset_config.has_labels = True

    elif data_split.startswith("val"):
        dataset_config.data_split_dir = "training"

        # Don't load labels for val split when running in test mode
        if eval_mode == "val":
            dataset_config.has_labels = True
        elif eval_mode == "test":
            dataset_config.has_labels = False

    elif data_split == "test":
        dataset_config.data_split_dir = "testing"
        dataset_config.has_labels = False

    else:
        raise ValueError("Invalid data split", data_split)

    # Convert to object to overwrite repeated fields
    dataset_config = config_builder.proto_to_obj(dataset_config)

    # Remove augmentation during evaluation
    dataset_config.aug_list = []

    # Build the dataset object
    dataset = DatasetBuilder.build_kitti_dataset(dataset_config, use_defaults=False)

    # Setup the model
    model_name = model_config.model_name

    # Convert to object to overwrite repeated fields
    model_config = config_builder.proto_to_obj(model_config)

    # Switch path drop off during evaluation
    model_config.path_drop_probabilities = [1.0, 1.0]

    with tf.Graph().as_default():
        if model_name == "rcnn_model":
            model = RcnnModel(
                model_config,
                train_val_test=eval_mode,
                dataset=dataset,
                batch_size=eval_config.batch_size,
            )
        elif model_name == "rpn_model":
            model = RpnModel(
                model_config,
                train_val_test=eval_mode,
                dataset=dataset,
                batch_size=eval_config.batch_size,
            )
        else:
            raise ValueError("Invalid model name {}".format(model_name))

        model_evaluator = Evaluator(model, dataset_config, eval_config)

        if kitti_native_eval_only:
            model_evaluator.run_kitti_native_eval(kitti_native_eval_step)
        elif evaluate_repeatedly:
            model_evaluator.repeated_checkpoint_run()
        else:
            model_evaluator.run_latest_checkpoints()


def main(_):
    parser = argparse.ArgumentParser()

    default_pipeline_config_path = hf.root_dir() + "/configs/rcnn_cars_example.config"

    parser.add_argument(
        "--pipeline_config",
        type=str,
        dest="pipeline_config_path",
        default=default_pipeline_config_path,
        help="Path to the pipeline config",
    )

    parser.add_argument(
        "--data_split",
        type=str,
        dest="data_split",
        default="val",
        help="Data split for evaluation",
    )

    parser.add_argument(
        "--save_rpn_feature",
        action="store_true",
        default=False,
        help="save features for separately rcnn training and evaluation",
    )

    parser.add_argument(
        "--for_rcnn_train",
        action="store_true",
        default=False,
        help="for separately rcnn training or evaluation, different NMS size used",
    )

    parser.add_argument(
        "--device", type=str, dest="device", default="0", help="CUDA device id"
    )

    args = parser.parse_args()

    # Parse pipeline config
    model_config, _, eval_config, dataset_config = config_builder.get_configs_from_pipeline_file(
        args.pipeline_config_path, is_training=False
    )

    # Overwrite data split
    dataset_config.data_split = args.data_split

    # Overwrite save_rpn_feature
    eval_config.save_rpn_feature = args.save_rpn_feature

    if model_config.model_name == "rpn_model":
        if args.for_rcnn_train:
            model_config.paths_config.pred_dir += "_for_rcnn_train"
        else:
            model_config.paths_config.pred_dir += "_for_rcnn_eval"
            model_config.rpn_config.rpn_train_pre_nms_size = (
                model_config.rpn_config.rpn_test_pre_nms_size
            )
            model_config.rpn_config.rpn_train_post_nms_size = (
                model_config.rpn_config.rpn_test_post_nms_size
            )
            model_config.rpn_config.rpn_train_nms_iou_thresh = (
                model_config.rpn_config.rpn_test_nms_iou_thresh
            )

    # Set CUDA device id
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    evaluate(model_config, eval_config, dataset_config)


if __name__ == "__main__":
    tf.app.run()
