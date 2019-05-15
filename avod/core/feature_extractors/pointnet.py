"""Contains modified PointCNN/XConv model definition to extract features from
PointCloud input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import avod.core.feature_extractors.tf_util as tf_util
from avod.core.feature_extractors.pointnet_util import (
    pointnet_sa_module,
    pointnet_sa_module_msg,
    pointnet_fp_module,
)

from avod.core.feature_extractors import pc_feature_extractor


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
                    print(sa_module.mlp.channel)
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
                    print(sa_msg_module.mlp)
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

            print("layer_points size: ", len(layer_points))

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

            # batch_size = points.get_shape()[0].value
            # num_point = points.get_shape()[1].value

            # Sampling and Grouping Layers
            # xconv_layers = self.config.xconv_layer
            # xconv_param_name = ("K", "D", "P", "C")
            # xconv_params = [
            #     dict(zip(xconv_param_name, xconv_param))
            #     for xconv_param in [
            #         xconv_layer.xconv_param for xconv_layer in xconv_layers
            #     ]
            # ]

            # layer_points, layer_features = [points], [features]
            # for layer_idx, layer_param in enumerate(xconv_params):
            #     print(layer_param)
            #     assert layer_param["D"] == 1, "Dilation must be 1 when use pointnet++"

            #     tag = "pointnet_sa_" + str(layer_idx + 1)

            #     nsample = layer_param["K"]
            #     npoint = layer_param["P"]
            #     # npoint = num_point if npoint == -1 else npoint
            #     nchannel = layer_param["C"]

            #     layer_point, layer_feature, _ = pointnet_sa_module(
            #         layer_points[-1],
            #         layer_features[-1],
            #         npoint=npoint,
            #         knn=True,
            #         radius=None,
            #         bn_decay=None,
            #         nsample=nsample,
            #         mlp=[nchannel // 2, nchannel // 2, nchannel],
            #         mlp2=None,
            #         group_all=False,
            #         is_training=is_training,
            #         scope=tag,
            #     )

            #     layer_points.append(layer_point)
            #     print("layer_point @ layer ", layer_idx)
            #     print(layer_points)
            #     print("layer_featrues @ layer ", layer_idx)
            #     layer_features.append(layer_feature)
            #     print(layer_features)

            # Feature Propagation Layers
            # xdconv_layers = self.config.xdconv_layer
            # xdconv_param_name = ("K", "D", "pts_layer_idx", "qrs_layer_idx")
            # xdconv_params = [
            #     dict(zip(xdconv_param_name, xdconv_param))
            #     for xdconv_param in [
            #         xdconv_layer.xdconv_param for xdconv_layer in xdconv_layers
            #     ]
            # ]

            # output_feature = None
            # for layer_idx, layer_param in enumerate(xdconv_params):
            #     print(layer_param)
            #     tag = "pointnet_fp" + str(layer_idx + 1)
            #     pts_layer_idx = layer_param["pts_layer_idx"]
            #     qrs_layer_idx = layer_param["qrs_layer_idx"]
            #     nchannel = xconv_params[qrs_layer_idx]["C"]

            #     # if pts_layer_idx != qrs_layer_idx + 1:
            #     #     print(layer_param, " skipped")
            #     #     continue

            #     sparser_point = layer_points[pts_layer_idx + 1]
            #     if layer_idx == 0:
            #         output_feature = layer_features[pts_layer_idx + 1]

            #     denser_point = layer_points[qrs_layer_idx + 1]
            #     denser_feature = layer_features[qrs_layer_idx + 1]

            #     output_feature = pointnet_fp_module(
            #         denser_point,
            #         sparser_point,
            #         denser_feature,
            #         output_feature,
            #         [nchannel, nchannel],
            #         is_training,
            #         bn_decay=None,
            #         scope=tag,
            #     )

            # comply with original pointnet structure
            # num_layer_points = len(layer_points)
            # for layer_idx in range(1, num_layer_points):
            #     tag = "pointnet_fp" + str(layer_idx)
            #     nchannel = xconv_params[num_layer_points-layer_idx-1]["C"]

            #     sparer_idx = num_layer_points - layer_idx
            #     denser_idx = num_layer_points - layer_idx - 1

            #     layer_features[denser_idx] = pointnet_fp_module(
            #                                      layer_points[denser_idx],
            #                                      layer_points[sparer_idx],
            #                                      layer_features[denser_idx],
            #                                      layer_features[sparer_idx],
            #                                      [nchannel,nchannel], is_training, bn_decay=None, scope=tag)

            # output_feature = layer_features[0]
            # print("output_feature")
            # print(output_feature)

            # for layer_point, layer_feature in zip(layer_points,)

            # Fully Connected Layers
            # fc_layers = self.config.fc_layer
            # for layer_idx, layer_param in enumerate(fc_layers):
            #     print(layer_param)
            #     nchannel = layer_param.C
            #     dropout_rate = layer_param.dropout_rate
            #     tag = "pointnet_fc" + str(layer_idx + 1)

            #     output_feature = tf_util.conv1d(
            #         output_feature,
            #         nchannel,
            #         1,
            #         padding="VALID",
            #         bn=True,
            #         is_training=is_training,
            #         scope=tag,
            #         bn_decay=None,
            #     )

            #     if layer_idx != len(fc_layers) - 1:
            #         output_feature = tf_util.dropout(
            #             output_feature,
            #             keep_prob=dropout_rate,
            #             is_training=is_training,
            #             scope=tag + "_dp",
            #         )

            # return points, output_feature

            # # Layer 1
            # l0_xyz, l0_featues = points, features
            # l1_xyz, l1_featues, l1_indices = pointnet_sa_module(l0_xyz, l0_featues, npoint=4096, radius=0.5, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=None, scope='layer1')
            # l2_xyz, l2_featues, l2_indices = pointnet_sa_module(l1_xyz, l1_featues, npoint=1024, radius=1.0, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=None, scope='layer2')
            # l3_xyz, l3_featues, l3_indices = pointnet_sa_module(l2_xyz, l2_featues, npoint=512, radius=2.0, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=None, scope='layer3')
            # l4_xyz, l4_featues, l4_indices = pointnet_sa_module(l3_xyz, l3_featues, npoint=216, radius=4.0, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=None, scope='layer4')

            # l3_featues = pointnet_fp_module(l3_xyz, l4_xyz, l3_featues, l4_featues, [256,256], is_training, None, scope='fa_layer1')
            # l2_featues = pointnet_fp_module(l2_xyz, l3_xyz, l2_featues, l3_featues, [256,256], is_training, None, scope='fa_layer2')
            # l1_featues = pointnet_fp_module(l1_xyz, l2_xyz, l1_featues, l2_featues, [256,128], is_training, None, scope='fa_layer3')
            # l0_featues = pointnet_fp_module(l0_xyz, l1_xyz, l0_featues, l1_featues, [128,128,128], is_training, None, scope='fa_layer4')

            # l0_xyz, l0_featues = points, features
            # l1_xyz, l1_featues = pointnet_sa_module_msg(l0_xyz, l0_featues, 4096, [0.1, 0.5], [16, 32], [[16, 16, 32], [32, 32, 64]], is_training, None, scope='layer1')
            # l2_xyz, l2_featues = pointnet_sa_module_msg(l1_xyz, l1_featues, 1024, [0.5, 1.0], [16, 32], [[64, 64, 128], [64, 96, 128]], is_training, None, scope='layer2')
            # l3_xyz, l3_featues = pointnet_sa_module_msg(l2_xyz, l2_featues, 256, [1.0, 2.0], [16, 32], [[128, 196, 256], [128, 196, 256]], is_training, None, scope='layer3')
            # l4_xyz, l4_featues = pointnet_sa_module_msg(l3_xyz, l3_featues, 64, [2.0, 4.0], [16, 32], [[256, 256, 512], [256, 384, 512]], is_training, None, scope='layer4')

            # # Feature Propagation layers
            # l3_featues = pointnet_fp_module(l3_xyz, l4_xyz, l3_featues, l4_featues, [1536, 512, 512], is_training, bn_decay=None, scope='fa_layer1')
            # l2_featues = pointnet_fp_module(l2_xyz, l3_xyz, l2_featues, l3_featues, [768, 512, 512], is_training, bn_decay=None, scope='fa_layer2')
            # l1_featues = pointnet_fp_module(l1_xyz, l2_xyz, l1_featues, l2_featues, [608, 256, 256], is_training, bn_decay=None, scope='fa_layer3')
            # l0_featues = pointnet_fp_module(l0_xyz, l1_xyz, l0_featues, l1_featues, [256, 128, 128], is_training, bn_decay=None, scope='fa_layer4')

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

            # net = tf_util.conv1d(l0_featues, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=None)
            # net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')

            return points, output_feature

            # with_X_transformation = self.config.with_X_transformation
            # sorting_method = self.config.sorting_method
            # B = tf.shape(points)[0]

            # if self.config.sampling == "fps":
            #     from sampling import tf_sampling

            # self.layer_pts = [points]
            # self.layer_fts = [features]
            # # XConv Layers
            # xconv_layers = self.config.xconv_layer
            # xconv_param_name = ("K", "D", "P", "C", "links")
            # xconv_params = [
            #     dict(zip(xconv_param_name, xconv_param))
            #     for xconv_param in [
            #         xconv_layer.xconv_param for xconv_layer in xconv_layers
            #     ]
            # ]
            # for layer_idx, layer_param in enumerate(xconv_params):
            #     print(layer_param)
            #     tag = "xconv_" + str(layer_idx + 1) + "_"
            #     K = layer_param["K"]
            #     D = layer_param["D"]
            #     P = layer_param["P"]
            #     C = layer_param["C"]
            #     # links = layer_param['links']
            #     links = []
            #     if self.config.sampling != "random" and links:
            #         print(
            #             "Error: flexible links are supported only when random sampling is used!"
            #         )
            #         exit()

            #     # sample P supporting points
            #     pts = self.layer_pts[-1]
            #     fts = self.layer_fts[-1]
            #     if P == -1 or (layer_idx > 0 and P == xconv_params[layer_idx - 1]["P"]):
            #         qrs = self.layer_pts[-1]
            #     else:
            #         if self.config.sampling == "fps":
            #             fps_indices = tf_sampling.farthest_point_sample(P, pts)
            #             batch_indices = tf.tile(
            #                 tf.reshape(tf.range(B), [-1, 1, 1]), (1, P, 1)
            #             )
            #             indices = tf.concat(
            #                 [batch_indices, tf.expand_dims(fps_indices, -1)], axis=-1
            #             )
            #             qrs = tf.gather_nd(pts, indices, name=tag + "qrs")  # (B, P, 3)
            #         elif self.config.sampling == "ids":
            #             indices = pf.inverse_density_sampling(pts, K, P)
            #             qrs = tf.gather_nd(pts, indices)
            #         elif self.config.sampling == "random":
            #             qrs = tf.slice(
            #                 pts, (0, 0, 0), (-1, P, -1), name=tag + "qrs"
            #             )  # (B, P, 3)
            #         else:
            #             print("Unknown sampling method!")
            #             exit()
            #     self.layer_pts.append(qrs)

            #     # xconv
            #     if layer_idx == 0:
            #         C_pts_fts = C // 2 if fts is None else C // 4
            #         depth_multiplier = 4
            #     else:
            #         C_prev = xconv_params[layer_idx - 1]["C"]
            #         C_pts_fts = C_prev // 4
            #         depth_multiplier = math.ceil(C / C_prev)
            #     with_global = (
            #         self.config.with_global and layer_idx == len(xconv_params) - 1
            #     )
            #     fts_xconv = xconv(
            #         pts,
            #         fts,
            #         qrs,
            #         tag,
            #         B,
            #         K,
            #         D,
            #         P,
            #         C,
            #         C_pts_fts,
            #         is_training,
            #         with_X_transformation,
            #         depth_multiplier,
            #         sorting_method,
            #         with_global,
            #     )
            #     fts_list = []
            #     for link in links:
            #         fts_from_link = self.layer_fts[link]
            #         if fts_from_link is not None:
            #             fts_slice = tf.slice(
            #                 fts_from_link,
            #                 (0, 0, 0),
            #                 (-1, P, -1),
            #                 name=tag + "fts_slice_" + str(-link),
            #             )
            #             fts_list.append(fts_slice)
            #     if fts_list:
            #         fts_list.append(fts_xconv)
            #         self.layer_fts.append(
            #             tf.concat(fts_list, axis=-1, name=tag + "fts_list_concat")
            #         )
            #     else:
            #         self.layer_fts.append(fts_xconv)

            # # XDConv Layers
            # xdconv_layers = self.config.xdconv_layer
            # xdconv_param_name = ("K", "D", "pts_layer_idx", "qrs_layer_idx")
            # xdconv_params = [
            #     dict(zip(xdconv_param_name, xdconv_param))
            #     for xdconv_param in [
            #         xdconv_layer.xdconv_param for xdconv_layer in xdconv_layers
            #     ]
            # ]
            # for layer_idx, layer_param in enumerate(xdconv_params):
            #     print(layer_param)
            #     tag = "xdconv_" + str(layer_idx + 1) + "_"
            #     K = layer_param["K"]
            #     D = layer_param["D"]
            #     pts_layer_idx = layer_param["pts_layer_idx"]
            #     qrs_layer_idx = layer_param["qrs_layer_idx"]

            #     pts = self.layer_pts[pts_layer_idx + 1]
            #     fts = (
            #         self.layer_fts[pts_layer_idx + 1]
            #         if layer_idx == 0
            #         else self.layer_fts[-1]
            #     )
            #     qrs = self.layer_pts[qrs_layer_idx + 1]
            #     fts_qrs = self.layer_fts[qrs_layer_idx + 1]
            #     P = xconv_params[qrs_layer_idx]["P"]
            #     C = xconv_params[qrs_layer_idx]["C"]
            #     C_prev = xconv_params[pts_layer_idx]["C"]
            #     C_pts_fts = C_prev // 4
            #     depth_multiplier = 1
            #     fts_xdconv = xconv(
            #         pts,
            #         fts,
            #         qrs,
            #         tag,
            #         B,
            #         K,
            #         D,
            #         P,
            #         C,
            #         C_pts_fts,
            #         is_training,
            #         with_X_transformation,
            #         depth_multiplier,
            #         sorting_method,
            #     )
            #     fts_concat = tf.concat(
            #         [fts_xdconv, fts_qrs], axis=-1, name=tag + "fts_concat"
            #     )
            #     fts_fuse = pf.dense(fts_concat, C, tag + "fts_fuse", is_training)
            #     self.layer_pts.append(qrs)
            #     self.layer_fts.append(fts_fuse)

            # self.fc_layers = [self.layer_fts[-1]]
            # fc_layers = self.config.fc_layer
            # for layer_idx, layer_param in enumerate(fc_layers):
            #     print(layer_param)
            #     C = layer_param.C
            #     dropout_rate = layer_param.dropout_rate
            #     fc = pf.dense(
            #         self.fc_layers[-1], C, "fc{:d}".format(layer_idx), is_training
            #     )
            #     fc_drop = tf.layers.dropout(
            #         fc,
            #         dropout_rate,
            #         training=is_training,
            #         name="fc{:d}_drop".format(layer_idx),
            #     )
            #     self.fc_layers.append(fc_drop)

            # return self.layer_pts[-1], self.fc_layers[-1]
