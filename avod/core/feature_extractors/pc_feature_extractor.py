from abc import abstractmethod

import tensorflow as tf
import random
from avod.core import pointfly as pf


class PcFeatureExtractor:
    def __init__(self, extractor_config):
        self.config = extractor_config

    def preprocess_input(self, pc_input_batches, input_config, is_training):
        """Preprocesses the given input. Split the point cloud input to
        pts(x,y,z) and fts(intensity) if necessary."""
        pts = pc_input_batches
        if input_config.pc_data_dim != 4:
            AssertionError("pc_data_dim must be 4 since" " intensity value is required")

        pts, fts = tf.split(pts, [3, 1], axis=-1, name="split_point_cloud_input")
        return pts, fts

    @abstractmethod
    def build(self, **kwargs):
        pass
