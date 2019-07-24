"""Contains modified PointCNN/XConv model definition to extract features from
PointCloud input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import hf.core.feature_extractors.tf_util as tf_util
from hf.core.feature_extractors.pointnet_util import (
    pointnet_sa_module,
    pointnet_sa_module_msg,
    pointnet_fp_module,
)

from hf.core.feature_extractors import pc_feature_extractor


class PointNet(pc_feature_extractor.PcFeatureExtractor):
    def build(self, points, features, is_training, scope="pc_pointnet"):

        with tf.variable_scope(scope):

            # SA/SA_MSG Module
            use_knn = self.config.use_knn
            use_sa_msg_module = self.config.use_sa_msg_module
            pointcnn_like_structure = self.config.pointcnn_like_structure

            layer_points, layer_features = [points], [features]

            if not use_sa_msg_module:
                for layer_idx, sa_module in enumerate(self.config.sa_module):
                    print(sa_module)
                    tag = "pointnet_sa_" + str(layer_idx + 1)
                    layer_point, layer_feature, _ = pointnet_sa_module(
                        layer_points[-1],
                        layer_features[-1],
                        npoint=sa_module.npoint,
                        knn=use_knn,
                        radius=sa_module.radius,
                        bn_decay=None,
                        nsample=sa_module.nsample,
                        mlp=sa_module.mlp.channel,
                        mlp2=None,
                        group_all=False,
                        is_training=is_training,
                        scope=tag,
                    )

                    layer_points.append(layer_point)
                    layer_features.append(layer_feature)
            else:
                for layer_idx, sa_msg_module in enumerate(self.config.sa_msg_module):
                    print(sa_msg_module)
                    mlps = []
                    for mlp in sa_msg_module.mlp:
                        mlps.append(mlp.channel)
                    tag = "pointnet_sa_msg_" + str(layer_idx + 1)
                    layer_point, layer_feature = pointnet_sa_module_msg(
                        layer_points[-1],
                        layer_features[-1],
                        npoint=sa_msg_module.npoint,
                        radius_list=sa_msg_module.radius,
                        nsample_list=sa_msg_module.nsample,
                        mlp_list=mlps,
                        is_training=is_training,
                        bn_decay=None,
                        scope=tag,
                    )

                    layer_points.append(layer_point)
                    layer_features.append(layer_feature)

            # Feature Propagation Layers
            output_feature = None
            if pointcnn_like_structure:
                for layer_idx, fp_module in enumerate(self.config.fp_module):
                    print(fp_module)
                    tag = "pointnet_fp_" + str(layer_idx + 1)
                    pts_layer_idx = fp_module.pts_layer_idx
                    qrs_layer_idx = fp_module.qrs_layer_idx

                    sparser_point = layer_points[pts_layer_idx + 1]
                    if layer_idx == 0:
                        output_feature = layer_features[pts_layer_idx + 1]

                    denser_point = layer_points[qrs_layer_idx + 1]
                    denser_feature = layer_features[qrs_layer_idx + 1]

                    # output feature is dense with respect to last layer,
                    # but sparse with respect to this layer
                    output_feature = pointnet_fp_module(
                        denser_point,
                        sparser_point,
                        denser_feature,
                        output_feature,
                        fp_module.mlp.channel,
                        is_training,
                        bn_decay=None,
                        scope=tag,
                    )
            else:
                # comply with original pointnet structure in paper
                num_layer_points = len(layer_points)
                assert len(self.config.fp_module) + 1 == num_layer_points
                for layer_idx, fp_module in enumerate(self.config.fp_module):
                    tag = "pointnet_fp" + str(layer_idx + 1)
                    sparer_idx = num_layer_points - layer_idx - 1
                    denser_idx = num_layer_points - layer_idx - 2

                    layer_features[denser_idx] = pointnet_fp_module(
                        layer_points[denser_idx],
                        layer_points[sparer_idx],
                        layer_features[denser_idx],
                        layer_features[sparer_idx],
                        fp_module.mlp.channel,
                        is_training,
                        bn_decay=None,
                        scope=tag,
                    )

                output_feature = layer_features[0]

            # Fully Connected Layers
            fc_layers = self.config.fc_layer
            for layer_idx, layer_param in enumerate(fc_layers):
                print(layer_param)
                nchannel = layer_param.C
                dropout_rate = layer_param.dropout_rate
                tag = "pointnet_fc" + str(layer_idx + 1)

                output_feature = tf_util.conv1d(
                    output_feature,
                    nchannel,
                    1,
                    padding="VALID",
                    bn=True,
                    is_training=is_training,
                    scope=tag,
                    bn_decay=None,
                )

                if layer_idx != len(fc_layers) - 1:
                    output_feature = tf_util.dropout(
                        output_feature,
                        keep_prob=dropout_rate,
                        is_training=is_training,
                        scope=tag + "_dp",
                    )

            return points, output_feature
