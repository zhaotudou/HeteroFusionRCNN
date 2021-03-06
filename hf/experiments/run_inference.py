"""Detection model inferencing.

This runs the DetectionModel evaluator in test mode to output detections.
"""

import argparse
import os
import sys

import tensorflow as tf

import hf
import hf.builders.config_builder_util as config_builder
from hf.builders.dataset_builder import DatasetBuilder
from hf.core.models.rcnn_model import RcnnModel
from hf.core.models.rpn_model import RpnModel
from hf.core.evaluator import Evaluator


def inference(model_config, eval_config, dataset_config, ckpt_indices):

    # Overwrite the defaults
    dataset_config = config_builder.proto_to_obj(dataset_config)

    dataset_config.data_split_dir = "training"
    if dataset_config.data_split == "test":
        dataset_config.data_split_dir = "testing"

    eval_config.eval_mode = "test"
    eval_config.evaluate_repeatedly = False

    dataset_config.has_labels = False
    # Enable this to see the actually memory being used
    eval_config.allow_gpu_mem_growth = True

    eval_config = config_builder.proto_to_obj(eval_config)
    # Grab the checkpoint indices to evaluate
    eval_config.ckpt_indices = ckpt_indices

    # Remove augmentation during evaluation in test mode
    dataset_config.aug_list = []

    # Build the dataset object
    dataset = DatasetBuilder.build_kitti_dataset(dataset_config, use_defaults=False)

    # Setup the model
    model_name = model_config.model_name
    # Overwrite repeated field
    model_config = config_builder.proto_to_obj(model_config)
    # Switch path drop off during evaluation
    model_config.path_drop_probabilities = [1.0, 1.0]

    with tf.Graph().as_default():
        if model_name == "rcnn_model":
            if eval_config.save_rpn_feature:
                raise ValueError("Full Model don't support --save_rpn_feature")
            model = RcnnModel(
                model_config,
                train_val_test=eval_config.eval_mode,
                dataset=dataset,
                batch_size=eval_config.batch_size,
            )
        elif model_name == "rpn_model":
            model = RpnModel(
                model_config,
                train_val_test=eval_config.eval_mode,
                dataset=dataset,
                batch_size=eval_config.batch_size,
            )
        else:
            raise ValueError("Invalid model name {}".format(model_name))

        model_evaluator = Evaluator(model, dataset_config, eval_config)
        model_evaluator.run_latest_checkpoints()


def main(_):
    parser = argparse.ArgumentParser()

    # Example usage
    # --checkpoint_name='rcnn_cars_example'
    # --data_split='test'
    # --ckpt_indices=50 100 112
    # Optional arg:
    # --device=0

    parser.add_argument(
        "--checkpoint_name",
        type=str,
        dest="checkpoint_name",
        required=True,
        help="Checkpoint name must be specified as a str\
                        and must match the experiment config file name.",
    )

    parser.add_argument(
        "--data_split",
        type=str,
        dest="data_split",
        required=True,
        help="Data split must be specified e.g. val or test",
    )

    parser.add_argument(
        "--ckpt_indices",
        type=int,
        nargs="+",
        dest="ckpt_indices",
        required=True,
        help="Checkpoint indices must be a set of \
        integers with space in between -> 0 10 20 etc",
    )

    parser.add_argument(
        "--save_rpn_feature",
        action="store_true",
        default=False,
        help="save features for separately rcnn training and evaluation",
    )

    parser.add_argument(
        "--device", type=str, dest="device", default="0", help="CUDA device id"
    )

    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    experiment_config = args.checkpoint_name + ".config"

    # Read the config from the experiment folder
    experiment_config_path = (
        hf.root_dir()
        + "/data/outputs/"
        + args.checkpoint_name
        + "/"
        + experiment_config
    )

    model_config, _, eval_config, dataset_config = config_builder.get_configs_from_pipeline_file(
        experiment_config_path, is_training=False
    )

    # Overwrite data split
    dataset_config.data_split = args.data_split

    # Overwrite save_rpn_feature
    eval_config.save_rpn_feature = args.save_rpn_feature

    # Set CUDA device id
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    inference(model_config, eval_config, dataset_config, args.ckpt_indices)


if __name__ == "__main__":
    tf.app.run()
