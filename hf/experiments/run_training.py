"""Detection model trainer.

This runs the DetectionModel trainer.
"""

import argparse
import os
import datetime

import tensorflow as tf
import horovod.tensorflow as hvd

import hf
import hf.builders.config_builder_util as config_builder
from hf.builders.dataset_builder import DatasetBuilder
from hf.core.models.rcnn_model import RcnnModel
from hf.core.models.rpn_model import RpnModel
from hf.core import trainer

tf.logging.set_verbosity(tf.logging.ERROR)


def train(model_config, train_config, dataset_config):

    dataset = DatasetBuilder.build_kitti_dataset(dataset_config, use_defaults=False)

    train_val_test = "train"
    model_name = model_config.model_name

    with tf.Graph().as_default():
        if model_name == "rpn_model":
            model = RpnModel(
                model_config,
                train_val_test=train_val_test,
                dataset=dataset,
                batch_size=train_config.batch_size,
            )
        elif model_name == "rcnn_model":
            model = RcnnModel(
                model_config,
                train_val_test=train_val_test,
                dataset=dataset,
                batch_size=train_config.batch_size,
            )
        else:
            raise ValueError("Invalid model_name")

        trainer.train(model, train_config)


def main(_):
    parser = argparse.ArgumentParser()

    # Defaults
    default_pipeline_config_path = (
        hf.root_dir() + "/configs/rpn_cars_pointcnn_paper.config"
    )
    default_data_split = "train"

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
        default=default_data_split,
        help="Data split for training",
    )

    args = parser.parse_args()

    # Parse pipeline config
    model_config, train_config, _, dataset_config = config_builder.get_configs_from_pipeline_file(
        args.pipeline_config_path, is_training=True
    )

    # Overwrite data split
    dataset_config.data_split = args.data_split

    hvd.init()
    print(
        "Rank {} training started at: {}".format(
            hvd.rank(), str(datetime.datetime.now())
        )
    )
    train(model_config, train_config, dataset_config)
    print(
        "Rank {} training finished at: {}".format(
            hvd.rank(), str(datetime.datetime.now())
        )
    )


if __name__ == "__main__":
    tf.app.run()
