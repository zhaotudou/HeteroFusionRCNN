from abc import abstractmethod

import tensorflow as tf
import random
from avod.core import pointfly as pf

class PcFeatureExtractor:

    def __init__(self, extractor_config):
        self.config = extractor_config

    def preprocess_input(self, pc_input_batches, input_config, is_training):
        """Preprocesses the given input.
        """
        self._input_config = input_config
        pc_data_dim = input_config.pc_data_dim
        use_extra_features = input_config.use_extra_features
        with_normal_feature = input_config.with_normal_feature
        
        if pc_data_dim > 3:
            pts, fts = tf.split(pc_input_batches, [3, pc_data_dim - 3], axis=-1,
                                name='split_points_features')
            if not use_extra_features:
                features_sampled = None
        else:
            pts = pc_input_batches
            fts = None
        ''' 
        if is_training: 
            # Augment
            points_augmented = pf.augment(points_sampled, xforms, jitter_range)
            features_augmented = features_sampled
            if features_sampled and with_normal_feature:
                if pc_data_dim < 6:
                    print('Only 3D normals are supported!')
                    exit()
                elif pc_data_dim == 6:
                    features_augmented = pf.augment(features_sampled, rotations)
                else:
                    normals, rest = tf.split(features_sampled, [3, pc_data_dim - 6])
                    normals_augmented = pf.augment(normals, rotations)
                    features_augmented = tf.concat([normals_augmented, rest], axis=-1)
            return points_augmented, features_augmented
        else:
            return points_sampled, features_sampled
        '''
        return pts, fts
    
    @abstractmethod
    def build(self, **kwargs):
        pass
