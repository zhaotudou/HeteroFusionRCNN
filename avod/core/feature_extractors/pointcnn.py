"""Contains modified PointCNN/XConv model definition to extract features from
PointCloud input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from avod.core import pointfly as pf
from avod.core.feature_extractors import pc_feature_extractor


def xconv(pts, fts, qrs, tag, B, K, D, P, C, C_pts_fts, is_training, with_X_transformation, depth_multiplier, sorting_method=None, with_global=False):
    '''Xconv, the basic operation block of PointCNN. This implements the Algorithm 1 in paper. 
    For a sampled representative point p, its k-nearest neighbors are P. The features associated
    with P is F, and Kel is the kernel as in typical convolution. 
    Input: Kel, p, P, F
    Output: F_p
      1) P`      <-  P-p               Transform to local coordinates
      2) F_delta <-  MLP_delta(P`)     Lift point into higher dimensional space 
      3) F_*     <-  [F_delta, F]   
      4) X       <-  MLP(P`)           Learn the X-transformation matrix
      5) F_X     <-  X * F_*           Weight and permute with the learnt X
      6) F_p     <-  Conv(Kel, F_X)      Typical convolution 

    Inputs:
       pts: (B, N, 3). The whole set of input points
       fts: (B, N, C). The whole set of input features
       qrs: (B, P, 3). The query points. They are representatives of the input points
       tag: string. Used for variable scope.
       B: int. batch size
       K: int. number of k-nearest neighbors
       D: int. dilation rate used for finding nearest neighbors
       P: int. number of query points for each batch slice
       C: int. number of channels for output features.
       C_pts_ft: int. number of channels for features from points. This is only used in
                 intermediate steps to lift point into higher dimensional space.
       is_training: bool
       with_X_transformation: bool. If true, a T-net (as in PointNet paper) will be used
          to "align all input set to a canonical space".
       depth_multiplier: depth multiplier used in separable_conv2d. 
       sorting_method: ?
       with_global: bool. If true, output another feature branch (qrs as input) 
           with two fully connected layers will be concantenated with the 
           final result. Thus changes the dimension of the final feature.
           Currently, this will only be true for the last layer of the PointNet encoder.
    Returns:
       fts_conv_3d: (B, P, C) when with_global is false,
                    (B, P, C + C//4) when with_global is true
                     
    '''
    # Get k-nearest points
    _, indices_dilated = pf.knn_indices_general(qrs, pts, K * D, True)
    indices = indices_dilated[:, :, ::D, :] # (B, P, K, 2)

    if sorting_method:
        indices = pf.sort_points(pts, indices, sorting_method)

    # 1) Transform to local coordinates 
    # P` <- P-p 
    nn_pts = tf.gather_nd(pts, indices, name=tag + 'nn_pts')  # (B, P, K, 3) 
    nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')  # (B, P, 1, 3)
    nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local')  # (B, P, K, 3)

    # 2) Lift point into higher dimensional space 
    # F_delta <- MLP_delta(P`)
    nn_fts_from_pts_0 = pf.dense(nn_pts_local, C_pts_fts, tag + 'nn_fts_from_pts_0', is_training) # (B, P, K, C_pts_fts)
    nn_fts_from_pts = pf.dense(nn_fts_from_pts_0, C_pts_fts, tag + 'nn_fts_from_pts', is_training) # (B, P, K, C_pts_fts)
    if fts is None:
        nn_fts_input = nn_fts_from_pts
    else:
        # 3) F_* <- [F_delta, F]
        nn_fts_from_prev = tf.gather_nd(fts, indices, name=tag + 'nn_fts_from_prev') # (B, P, K, C)
        nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1, name=tag + 'nn_fts_input') # (B, P, K, C + C_pts_fts)

    
    if with_X_transformation:
        # 4) Learn the X-transformation matrix
        # X <- MLP(P`)
        X_0 = pf.conv2d(nn_pts_local, K * K, tag + 'X_0', is_training, (1, K))  # (B, P, 1, K*K)
        X_0_KK = tf.reshape(X_0, [B, P, K, K], name=tag + 'X_0_KK')             # (B, P, K, K)
        X_1 = pf.depthwise_conv2d(X_0_KK, K, tag + 'X_1', is_training, (1, K))  # (B, P, 1, K*K)
        X_1_KK = tf.reshape(X_1, [B, P, K, K], name=tag + 'X_1_KK')             # (B, P, K, K))
        X_2 = pf.depthwise_conv2d(X_1_KK, K, tag + 'X_2', is_training, (1, K), activation=None) # (B, P, 1, K*K)
        X_2_KK = tf.reshape(X_2, (B, P, K, K), name=tag + 'X_2_KK') # (B, P, K, K)
        # 5) Weight and permute with the learnt X
        # F_X  <- X * F_*
        fts_X = tf.matmul(X_2_KK, nn_fts_input, name=tag + 'fts_X') # (B, P, K, C_pts_fts)
    else:
        fts_X = nn_fts_input    # (B, P, K, C_pts_fts)
    
    # 6) Typical convolution 
    # F_p <- Conv(K, F_X)
    fts_conv = pf.separable_conv2d(fts_X, C, tag + 'fts_conv', 
                      is_training, (1, K), depth_multiplier=depth_multiplier) # (B, P, 1, C)
    fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=tag + 'fts_conv_3d') # (B, P, C)

    if with_global:
        fts_global_0 = pf.dense(qrs, C // 4, tag + 'fts_global_0', is_training) # (B, P, C//4)
        fts_global = pf.dense(fts_global_0, C // 4, tag + 'fts_global', is_training) # (B, P, C//4)
        return tf.concat([fts_global, fts_conv_3d], axis=-1, name=tag + 'fts_conv_3d_with_global') # (B, P, C + C//4)
    else:
        return fts_conv_3d # (B, P, C)

class PointCNN(pc_feature_extractor.PcFeatureExtractor):

    def build(self, points, features, is_training, scope='pc_pointcnn'):
        
        with tf.variable_scope(scope):
            with_X_transformation = self.config.with_X_transformation
            sorting_method = self.config.sorting_method
            B = tf.shape(points)[0]

            if self.config.sampling == 'fps':
                from sampling import tf_sampling

            self.layer_pts = [points]
            self.layer_fts = [features]
            # XConv Layers
            xconv_layers = self.config.xconv_layer
            xconv_param_name = ('K', 'D', 'P', 'C', 'links')
            xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in 
                            [xconv_layer.xconv_param for xconv_layer in xconv_layers]]
            for layer_idx, layer_param in enumerate(xconv_params):
                print(layer_param)
                tag = 'xconv_' + str(layer_idx + 1) + '_'
                K = layer_param['K']
                D = layer_param['D']
                P = layer_param['P']
                C = layer_param['C']
                #links = layer_param['links']
                links = []
                if self.config.sampling != 'random' and links:
                    print('Error: flexible links are supported only when random sampling is used!')
                    exit()

                # sample P supporting points
                pts = self.layer_pts[-1]
                fts = self.layer_fts[-1]
                if P == -1 or (layer_idx > 0 and P == xconv_params[layer_idx - 1]['P']):
                    qrs = self.layer_pts[-1]
                else:
                    if self.config.sampling == 'fps':
                        fps_indices = tf_sampling.farthest_point_sample(P, pts)
                        batch_indices = tf.tile(tf.reshape(tf.range(B), [-1, 1, 1]), (1, P, 1))
                        indices = tf.concat([batch_indices, tf.expand_dims(fps_indices,-1)], axis=-1)
                        qrs = tf.gather_nd(pts, indices, name= tag + 'qrs') # (B, P, 3)
                    elif self.config.sampling == 'ids':
                        indices = pf.inverse_density_sampling(pts, K, P)
                        qrs = tf.gather_nd(pts, indices)
                    elif self.config.sampling == 'random':
                        qrs = tf.slice(pts, (0, 0, 0), (-1, P, -1), name=tag + 'qrs')  # (B, P, 3)
                    else:
                        print('Unknown sampling method!')
                        exit()
                self.layer_pts.append(qrs)

                # xconv
                if layer_idx == 0:
                    C_pts_fts = C // 2 if fts is None else C // 4
                    depth_multiplier = 4
                else:
                    C_prev = xconv_params[layer_idx - 1]['C']
                    C_pts_fts = C_prev // 4
                    depth_multiplier = math.ceil(C / C_prev)
                with_global = (self.config.with_global and layer_idx == len(xconv_params) - 1)
                fts_xconv = xconv(pts, fts, qrs, tag, B, K, D, P, C, C_pts_fts, is_training, with_X_transformation,
                                  depth_multiplier, sorting_method, with_global)
                fts_list = []
                for link in links:
                    fts_from_link = self.layer_fts[link]
                    if fts_from_link is not None:
                        fts_slice = tf.slice(fts_from_link, (0, 0, 0), (-1, P, -1), name=tag + 'fts_slice_' + str(-link))
                        fts_list.append(fts_slice)
                if fts_list:
                    fts_list.append(fts_xconv)
                    self.layer_fts.append(tf.concat(fts_list, axis=-1, name=tag + 'fts_list_concat'))
                else:
                    self.layer_fts.append(fts_xconv)
            
            # XDConv Layers
            xdconv_layers = self.config.xdconv_layer
            xdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
            xdconv_params = [dict(zip(xdconv_param_name, xdconv_param)) for xdconv_param in 
                                    [xdconv_layer.xdconv_param for xdconv_layer in xdconv_layers]]
            for layer_idx, layer_param in enumerate(xdconv_params):
                print(layer_param)
                tag = 'xdconv_' + str(layer_idx + 1) + '_'
                K = layer_param['K']
                D = layer_param['D']
                pts_layer_idx = layer_param['pts_layer_idx']
                qrs_layer_idx = layer_param['qrs_layer_idx']

                pts = self.layer_pts[pts_layer_idx + 1]
                fts = self.layer_fts[pts_layer_idx + 1] if layer_idx == 0 else self.layer_fts[-1]
                qrs = self.layer_pts[qrs_layer_idx + 1]
                fts_qrs = self.layer_fts[qrs_layer_idx + 1]
                P = xconv_params[qrs_layer_idx]['P']
                C = xconv_params[qrs_layer_idx]['C']
                C_prev = xconv_params[pts_layer_idx]['C']
                C_pts_fts = C_prev // 4
                depth_multiplier = 1
                fts_xdconv = xconv(pts, fts, qrs, tag, B, K, D, P, C, C_pts_fts, is_training, with_X_transformation,
                                   depth_multiplier, sorting_method)
                fts_concat = tf.concat([fts_xdconv, fts_qrs], axis=-1, name=tag + 'fts_concat')
                fts_fuse = pf.dense(fts_concat, C, tag + 'fts_fuse', is_training)
                self.layer_pts.append(qrs)
                self.layer_fts.append(fts_fuse)

            self.fc_layers = [self.layer_fts[-1]]
            fc_layers = self.config.fc_layer
            for layer_idx, layer_param in enumerate(fc_layers):
                print(layer_param)
                C = layer_param.C
                dropout_rate = layer_param.dropout_rate
                fc = pf.dense(self.fc_layers[-1], C, 'fc{:d}'.format(layer_idx), is_training)
                fc_drop = tf.layers.dropout(fc, dropout_rate, training=is_training, name='fc{:d}_drop'.format(layer_idx))
                self.fc_layers.append(fc_drop)

            return self.layer_pts[-1], self.fc_layers[-1]
