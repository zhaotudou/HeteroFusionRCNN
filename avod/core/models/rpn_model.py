import numpy as np
import cv2
import random
import tensorflow as tf
from tensorflow.contrib import slim

from avod.builders import feature_extractor_builder
from avod.core.models import model_util
from avod.core import anchor_filter
from avod.core import anchor_projector
from avod.core import box_3d_encoder
from avod.core import box_3d_projector
from avod.core import bin_based_box3d_encoder
from avod.core import box_util
from avod.core import oriented_nms
from avod.core import constants
from avod.core import losses
from avod.core import model
from avod.core import projection
from avod.core import pointfly as pf
from avod.core import summary_utils
from avod.core.anchor_generators import grid_anchor_3d_generator
from avod.datasets.kitti import kitti_aug


class RpnModel(model.DetectionModel):
    ##############################
    # Keys for Placeholders
    ##############################
    PL_PC_INPUT = 'pc_input_pl'
    PL_LABEL_SEGS = 'label_segs_pl'
    PL_IMG_INPUT = 'img_input_pl'
    PL_CALIB_P2 = 'frame_calib_p2'

    ##############################
    # Keys for Predictions
    ##############################
    PRED_SEG_SOFTMAX = 'rpn_seg_softmax'
    PRED_SEG_GT = 'rpn_seg_gt'
    PRED_TOTAL_PTS = 'rpn_total_pts'
    
    PRED_FG_CLS = 'rpn_fg_cls'
    PRED_FG_REG = 'rpn_fg_reg'
    PRED_FG_CLS_GT = 'rpn_fg_cls_gt'
    PRED_FG_REG_GT = 'rpn_fg_reg_gt'
    PRED_FG_PTS = 'rpn_fg_pts_num'

    PRED_NMS_INDICES = 'rpn_nms_indices'
    PRED_PROPOSALS = 'rpn_proposals'
    PRED_OBJECTNESS_SOFTMAX = 'rpn_objectness_softmax'
    ##############################
    # Keys for Loss
    ##############################
    LOSS_RPN_SEGMENTATION = 'rpn_seg_loss'
    LOSS_RPN_BIN_CLASSIFICATION = 'rpn_cls_loss'
    LOSS_RPN_REGRESSION = 'rpn_reg_loss'

    def __init__(self, model_config, train_val_test, dataset, batch_size=2):
        """
        Args:
            model_config: configuration for the model
            train_val_test: "train", "val", or "test"
            dataset: the dataset that will provide samples and ground truth
        """

        # Sets model configs (_config)
        super(RpnModel, self).__init__(model_config)

        self._batch_size = batch_size
        
        if train_val_test not in ["train", "val", "test"]:
            raise ValueError('Invalid train_val_test value,'
                             'should be one of ["train", "val", "test"]')
        self._train_val_test = train_val_test

        self._is_training = (self._train_val_test == 'train')

        # Input config
        input_config = self._config.input_config
        self._pc_sample_pts = input_config.pc_sample_pts
        self._pc_data_dim = input_config.pc_data_dim
        self._pc_sample_pts_variance = input_config.pc_sample_pts_variance
        self._pc_sample_pts_clip = input_config.pc_sample_pts_clip
        self.NUM_FG_POINT = 2048

        self._img_h = input_config.img_dims_h
        self._img_w = input_config.img_dims_w
        self._img_depth = input_config.img_depth

        # Rpn config
        rpn_config = self._config.rpn_config
        self._proposal_roi_crop_size = rpn_config.rpn_proposal_roi_crop_size
        self._fusion_method = rpn_config.rpn_fusion_method

        if self._train_val_test in ["train", "val"]:
            self._nms_size = rpn_config.rpn_train_nms_size
            self._nms_iou_thresh = rpn_config.rpn_train_nms_iou_thresh
        else:
            self._nms_size = rpn_config.rpn_test_nms_size
            self._nms_iou_thresh = rpn_config.rpn_test_nms_iou_thresh
       
        self.S = rpn_config.rpn_xz_search_range
        self.DELTA = rpn_config.rpn_xz_bin_len
        self.NUM_BIN_X = int(2 * self.S / self.DELTA)
        self.NUM_BIN_Z = self.NUM_BIN_X
        
        self.R = rpn_config.rpn_theta_search_range * np.pi
        self.DELTA_THETA = 2 * self.R / rpn_config.rpn_theta_bin_num
        self.NUM_BIN_THETA = rpn_config.rpn_theta_bin_num

        # Feature Extractor Nets
        self._pc_feature_extractor = \
            feature_extractor_builder.get_extractor(
                self._config.layers_config.pc_feature_extractor)
        self._img_feature_extractor = \
            feature_extractor_builder.get_extractor(
                self._config.layers_config.img_feature_extractor)
        # Network input placeholders
        self.placeholders = dict()

        # Inputs to network placeholders
        self._placeholder_inputs = dict()

        # Information about the current sample
        self._samples_info = []

        # Dataset
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        # Overwrite the dataset's variable with the config
        self.dataset.train_val_test = self._train_val_test
        self._area_extents = self.dataset.kitti_utils.area_extents
        self._bev_extents = self.dataset.kitti_utils.bev_extents
        self._cluster_sizes, _ = self.dataset.get_cluster_info()
        self._anchor_strides = self.dataset.kitti_utils.anchor_strides
        self._anchor_generator = \
            grid_anchor_3d_generator.GridAnchor3dGenerator()
        
        self._path_drop_probabilities = self._config.path_drop_probabilities
        self._train_on_all_samples = self._config.train_on_all_samples
        self._eval_all_samples = self._config.eval_all_samples
        # Overwrite the dataset's variable with the config
        self.dataset.train_on_all_samples = self._train_on_all_samples
        self.dataset.eval_all_samples = self._eval_all_samples
        if self._train_val_test in ["val", "test"]:
            # Disable path-drop, this should already be disabled inside the
            # evaluator, but just in case.
            self._path_drop_probabilities[0] = 1.0
            self._path_drop_probabilities[1] = 1.0

    def _add_placeholder(self, dtype, shape, name):
        placeholder = tf.placeholder(dtype, shape, name)
        self.placeholders[name] = placeholder
        return placeholder

    def _set_up_input_pls(self):
        """Sets up input placeholders by adding them to self._placeholders.
        Keys are defined as self.PL_*.
        """
        with tf.variable_scope('pc_input'):
            # Placeholder for PC input, to be filled in with feed_dict
            pc_input_placeholder = self._add_placeholder(
                                        tf.float32, 
                                        (self._batch_size, None, self._pc_data_dim),
                                        self.PL_PC_INPUT) #(B,P',3)

            self._pc_pts_preprocessed, self._pc_fts_preprocessed = \
                self._pc_feature_extractor.preprocess_input(
                    pc_input_placeholder, 
                    self._config.input_config,
                    self._is_training)
        
        with tf.variable_scope('img_input'):
            # Take variable size input images
            img_input_placeholder = self._add_placeholder(
                tf.float32,
                [self._batch_size, self._img_h, self._img_w, self._img_depth],
                self.PL_IMG_INPUT)

            self._img_preprocessed = \
                self._img_feature_extractor.preprocess_input(img_input_placeholder)

        with tf.variable_scope('pl_labels'):
            self._add_placeholder(tf.float32, [self._batch_size, None, 8], 
                                  self.PL_LABEL_SEGS) #(B,P,8)
        
        with tf.variable_scope('sample_info'):
            # the calib matrix shape is (3 x 4)
            self._add_placeholder(tf.float32, [self._batch_size, 3, 4], 
                                  self.PL_CALIB_P2)

    def _set_up_feature_extractors(self):
        """Sets up feature extractors and stores feature maps and
        bottlenecks as member variables.
        """

        self._pc_pts, self._pc_fts = \
            self._pc_feature_extractor.build(
                self._pc_pts_preprocessed,
                self._pc_fts_preprocessed,
                self._is_training) #(B,P,3) (B,P,C)
        
        self._img_fts, _ = \
            self._img_feature_extractor.build(
                self._img_preprocessed,
                self._is_training) #(B,H,W,C1)


    def _gather_residuals(self, res_x_norms, res_z_norms, res_theta_norms,
                                bin_x, bin_z, bin_theta):
        
        '''
        Input:
            res_x_norms: (B,p,K)
            bin_x:(B,p)
        return:
            res_x_norm: (B,p)

        TF version: (if p is not None)
        ##########
        '''
        B = bin_x.shape[0].value
        p = bin_x.shape[1].value # maybe None
        Bs = tf.range(B)
        ps = tf.range(p)
        mB, mp = tf.meshgrid(Bs, ps)
        Bp = tf.stack([tf.transpose(mB), tf.transpose(mp)], axis=2) # (B,p,2)

        BpK_x = tf.concat([Bp, tf.reshape(bin_x, [B,p,1])], axis=2) # (B,p,3)
        res_x_norm = tf.gather_nd(res_x_norms, BpK_x) #(B,p)
        
        BpK_z = tf.concat([Bp, tf.reshape(bin_z, [B,p,1])], axis=2) # (B,p,3)
        res_z_norm = tf.gather_nd(res_z_norms, BpK_z) #(B,p)
        
        BpK_theta = tf.concat([Bp, tf.reshape(bin_theta, [B,p,1])], axis=2) # (B,p,3)
        res_theta_norm = tf.gather_nd(res_theta_norms, BpK_theta) #(B,p)
    

        '''
        NumPy version: if p is None, by using tf.py_func, p should be determined
        #############
        res_x_norm = np.take_along_axis(res_x_norms, np.expand_dims(bin_x, -1), axis=-1) #(B,p,1)
        res_x_norm = np.squeeze(res_x_norm, -1)
         
        res_z_norm = np.take_along_axis(res_z_norms, np.expand_dims(bin_z, -1), axis=-1) #(B,p,1)
        res_z_norm = np.squeeze(res_z_norm, -1)
        
        res_theta_norm = np.take_along_axis(res_theta_norms, np.expand_dims(bin_theta, -1), axis=-1) #(B,p,1)
        res_theta_norm = np.squeeze(res_theta_norm, -1)
        '''

        return res_x_norm, res_z_norm, res_theta_norm

    def _gather_mean_sizes(self, cluster_sizes, cls_preds):
        '''
        Input:
            cluster_sizes: (Klass, Cluster=1, 3) [l,w,h], Klass is 0-based
            cls_preds: (B,p), [klass], kclass is 1-based, 0-background
        Output
            mean_sizes: (B,p,3) [l,w,h]

        TF version: (if p is not None)
        ##########
        '''
        B = cls_preds.shape[0].value
        p = cls_preds.shape[1].value
        
        Bs = tf.range(B)
        ps = tf.range(p)
        mB, mp = tf.meshgrid(Bs, ps)
        Bp = tf.stack([tf.transpose(mB), tf.transpose(mp)], axis=2) # (B,p,2)

        K_mean_sizes = tf.reshape(cluster_sizes, [-1,3])
        pK_mean_sizes = tf.tile(tf.expand_dims(K_mean_sizes, 0), [p,1,1])
        BpK_mean_sizes = tf.tile(tf.expand_dims(pK_mean_sizes, 0), [B,1,1,1])

        BpK = tf.concat([Bp, tf.reshape(cls_preds, [B,p,1])], axis=2) # (B,p,3)
        
        mean_sizes = tf.gather_nd(BpK_mean_sizes, BpK)

        '''
        NumPy version: if p is None, by using tf.py_func, p should be determined
        #############
        K_mean_sizes = np.reshape(cluster_sizes, (-1,3))
        K_mean_sizes = np.vstack([np.asarray([0.0, 0.0, 0.0]), K_mean_sizes]) # insert 0-background
        mean_sizes = K_mean_sizes[cls_preds]
        
        return mean_sizes.astype(np.float32)
        '''
        
        return mean_sizes

    def _get_proj_indices(self, proj_pts2d):
        '''
        Input:
            proj_pts2d: (B,P,2) [u,v] int32
        Output:
            proj_indices: (B,P,3) [b,u,v] int32
        '''
        '''
        TF version: (if P is not None)
        ##########
        '''
        B = proj_pts2d.shape[0].value
        P = proj_pts2d.shape[1].value
        proj_indices = tf.concat(
                          [tf.expand_dims(tf.tile(tf.reshape(tf.range(0,B), [B,1]), [1,P]), axis=-1), #(B,P,1)
                           proj_pts2d],   #(B,P,2)
                          axis=-1)    #(B,P,3)
        '''
        NumPy version: if P is None, by using tf.py_func, P should be determined
        #############
        B = proj_pts2d.shape[0]
        P = proj_pts2d.shape[1]
        proj_indices = np.concatenate(
                        (np.expand_dims(np.tile(np.arange(0, B).reshape((B,1)), (1,P)), axis=-1),
                         proj_pts2d),
                        axis=-1)
        '''
        return proj_indices

    def build(self):

        # Setup input placeholders
        self._set_up_input_pls()

        # Setup feature extractors
        self._set_up_feature_extractors()
        
        # branch-1: foreground point segmentation
        #########################################
        with tf.variable_scope('foreground_segmentation'):
            #TODO: num seg class should be 2 (bkg/fg), not num_classes + 1
            seg_logits = pf.dense(self._pc_fts, self.num_classes + 1, 'seg_logits', 
                                  self._is_training, with_bn=False, activation=None)    #(B,P,K)
            seg_softmax = tf.nn.softmax(seg_logits, name='seg_softmax') #(B,P,K)
            seg_preds = tf.argmax(seg_softmax, axis=-1, name='seg_predictions', output_type=tf.int32) #(B,P)
            seg_scores = tf.reduce_max(seg_softmax[:,:,1:], axis=-1, name='seg_scores') #(B,P)
            
        label_segs = self.placeholders[self.PL_LABEL_SEGS] #(B,P,8)
        label_cls = label_segs[:,:,0]
        label_box_3d = label_segs[:,:,1:]
        
        # foreground point masking
        with tf.variable_scope('fg_masking'):
            if self._train_val_test in ['train', 'val']:
                self._fg_mask = label_cls > 0 #(B,P)
            else:
               self._fg_mask = seg_preds > 0 #(B,P)
            
            fg_indices = model_util.point_cloud_masking(self._fg_mask, self.NUM_FG_POINT) #(B,F,2)
            fg_pts = tf.reshape(tf.gather_nd(self._pc_pts, fg_indices),
                                [self._batch_size, self.NUM_FG_POINT, self._pc_pts.shape[2].value]) #(B,F,3)
            fg_fts = tf.reshape(tf.gather_nd(self._pc_fts, fg_indices),
                                [self._batch_size, self.NUM_FG_POINT, self._pc_fts.shape[2].value]) #(B,F,C+C1)
            fg_preds = tf.reshape(tf.gather_nd(seg_preds, fg_indices),
                                  [self._batch_size, self.NUM_FG_POINT])  #(B,F)
            fg_scores = tf.reshape(tf.gather_nd(seg_scores, fg_indices),
                                   [self._batch_size, self.NUM_FG_POINT]) #(B,F)
            fg_label_boxes_3d = tf.reshape(tf.gather_nd(label_box_3d, fg_indices),
                                           [self._batch_size, self.NUM_FG_POINT, 7]) #(B,F,7)

        # fuse fg pc + img features
        #########################################
        with tf.variable_scope('fts_fuse'):
            proj_pts2d = projection.tf_rect_to_image(fg_pts, self.placeholders[self.PL_CALIB_P2]) #(B,F,2)
            proj_pts2d = tf.cast(proj_pts2d, tf.int32) #(B,F,2)
            proj_indices = self._get_proj_indices(proj_pts2d) #(B,F,3)
            self._proj_indices = tf.gather(proj_indices, [0,2,1], axis=-1) # (B,F,3), image's shape is (y,x)
            proj_img_fts = tf.gather_nd(self._img_fts, self._proj_indices) # (B,F,C1)
            
            fusion_mean_div_factor = 2.0
            # If both img and pc probabilites are set to 1.0, don't do
            # path drop.
            if not (self._path_drop_probabilities[0] ==
                    self._path_drop_probabilities[1] == 1.0):
                with tf.variable_scope('rpn_path_drop'):

                    random_values = tf.random_uniform(shape=[3],
                                                      minval=0.0,
                                                      maxval=1.0)

                    img_mask, pc_mask = self.create_path_drop_masks(
                        self._path_drop_probabilities[0],
                        self._path_drop_probabilities[1],
                        random_values)

                    fg_fts = tf.multiply(fg_fts, pc_mask)
                    proj_img_fts = tf.multiply(proj_img_fts, img_mask)

                    self.img_path_drop_mask = img_mask
                    self.pc_path_drop_mask = pc_mask

                    # Overwrite the division factor
                    fusion_mean_div_factor = img_mask + pc_mask

            if self._fusion_method == 'mean':
                assert(self._pc_fts.shape[-1].value == self._img_fts.shape[-1].value)
                tf_features_sum = tf.add(fg_fts, proj_img_fts)
                fg_fuse_fts = tf.divide(tf_features_sum, fusion_mean_div_factor) #(B,F,C)
            elif self._fusion_method == 'concat':
                fg_fuse_fts = tf.concat([fg_fts, proj_img_fts], axis=-1) #(B,F,C+C1)
            else:
                raise ValueError('Invalid fusion method', self._fusion_method)

        # branch-2: bin-based 3D proposal generation
        #########################################
        with tf.variable_scope('bin_based_rpn'):
            # Parse rpn layers config
            fc_layers = [fg_fuse_fts]
            layers_config = self._config.layers_config.rpn_config.fc_layer
            for layer_idx, layer_param in enumerate(layers_config):
                C = layer_param.C
                dropout_rate = layer_param.dropout_rate
                fc = pf.dense(fc_layers[-1], C, 'fc{:d}'.format(layer_idx), self._is_training)
                fc_drop = tf.layers.dropout(fc, dropout_rate, training=self._is_training, 
                                            name='fc{:d}_drop'.format(layer_idx))
                fc_layers.append(fc_drop)
            
            fc_output = pf.dense(fc_layers[-1], 
                                 self.NUM_BIN_X*2 + self.NUM_BIN_Z*2 + self.NUM_BIN_THETA*2 + 4, 
                                 'fc_output', self._is_training, activation=None)
            bin_x_logits, res_x_norms, \
            bin_z_logits, res_z_norms, \
            bin_theta_logits, res_theta_norms, \
            res_y, res_size = self._parse_rpn_output(fc_output)
            res_y = tf.squeeze(res_y, [-1])

        with tf.variable_scope('histograms_rpn'):
            with tf.variable_scope('bin_based_proposal'):
                for fc_layer in fc_layers:
                    # fix the name to avoid tf warnings
                    tf.summary.histogram(fc_layer.name.replace(':', '_'),
                                         fc_layer)
        # Return the proposals
        ######################################################
        with tf.variable_scope('proposals'):
            # Decode bin-based 3D Box 
            bin_x = tf.argmax(bin_x_logits, axis=-1, output_type=tf.int32) #(B,F)
            bin_z = tf.argmax(bin_z_logits, axis=-1, output_type=tf.int32) #(B,F)
            bin_theta = tf.argmax(bin_theta_logits, axis=-1, output_type=tf.int32) #(B,F)
            
            res_x_norm, res_z_norm, res_theta_norm = self._gather_residuals(
                res_x_norms, res_z_norms, res_theta_norms, bin_x, bin_z, bin_theta)
            
            mean_sizes = self._gather_mean_sizes(
                tf.convert_to_tensor(np.asarray(self._cluster_sizes, dtype=np.float32)), fg_preds)
            
            with tf.variable_scope('decoding'):
                proposals = bin_based_box3d_encoder.tf_decode(
                        fg_pts, 0,
                        bin_x, res_x_norm,
                        bin_z, res_z_norm,
                        bin_theta, res_theta_norm,
                        res_y, res_size, mean_sizes,
                        self.S, self.DELTA, self.R, self.DELTA_THETA) # (B,F,7)
            
            # NMS
            if self._train_val_test == 'train':
                # to speed up training, skip NMS, as we don't care what top_* is during training
                print('Skip RPN-NMS during training')
                nms_indices = tf.zeros([self._batch_size, self._nms_size], tf.int32)
            else:
                # oriented-NMS is much slower than non-oriented-NMS (tf.image.non_max_suppression)
                # while get significant higher proposal recall@IoU=0.7
                oriented_NMS = True
                print("RPN oriented_NMS = " + str(oriented_NMS))
                # BEV projection
                with tf.variable_scope('bev_projection'):
                    def sb_project_to_bev_fn(sb_proposals):
                        if oriented_NMS:
                            sb_bev_proposal_boxes, _ = tf.py_func(
                                box_3d_projector.project_to_bev,
                                [sb_proposals, tf.constant(self._bev_extents)],
                                (tf.float32, tf.float32))
                        else:
                            # ortho rotating
                            sb_proposal_anchors = box_3d_encoder.tf_box_3d_to_anchor(
                                                            sb_proposals)
                            sb_bev_proposal_boxes, _ = anchor_projector.project_to_bev(
                                                                    sb_proposal_anchors, 
                                                                    self._bev_extents)
                            sb_bev_proposal_boxes = anchor_projector.reorder_projected_boxes(
                                                                sb_bev_proposal_boxes)
                        return sb_bev_proposal_boxes
            
                    bev_proposal_boxes = tf.map_fn(
                        sb_project_to_bev_fn,
                        elems=proposals,
                        dtype=tf.float32)
                    if oriented_NMS:
                        bev_proposal_boxes = tf.reshape(bev_proposal_boxes, 
                                                        [self._batch_size, -1, 4, 2])
                # BEV-NMS and ignore multiclass
                with tf.variable_scope('bev_nms'):
                    def sb_nms_fn(args):
                        (sb_boxes, sb_scores) = args
                        if oriented_NMS: 
                            sb_nms_indices = tf.py_func(
                                oriented_nms.nms,
                                [sb_boxes, sb_scores, 
                                tf.constant(self._nms_iou_thresh), tf.constant(self._nms_size)],
                                tf.int32)
                        else:
                            sb_nms_indices = tf.image.non_max_suppression(
                                sb_boxes,
                                sb_scores,
                                max_output_size=self._nms_size,
                                iou_threshold=self._nms_iou_thresh)
                        
                        sb_nms_indices = tf.pad(sb_nms_indices, 
                                            [[0, self._nms_size-tf.shape(sb_nms_indices)[0]]], mode='CONSTANT', constant_values=-1)
                        return sb_nms_indices
                    
                    bin_x_scores = tf.reduce_max(tf.nn.softmax(bin_x_logits), axis=-1)
                    bin_z_scores = tf.reduce_max(tf.nn.softmax(bin_z_logits), axis=-1)
                    bin_theta_scores = tf.reduce_max(tf.nn.softmax(bin_theta_logits), axis=-1)
                    confidence = fg_scores * bin_x_scores * bin_z_scores * bin_theta_scores
                    nms_indices = tf.map_fn(
                        sb_nms_fn,
                        elems=[bev_proposal_boxes, confidence],
                        dtype=tf.int32)
            
        ######################################################
        # GTs for the loss function & metrics
        ######################################################
        # Ground Truth Seg
        with tf.variable_scope('seg_one_hot_classes'):
            segs_gt_one_hot = tf.one_hot(
                tf.to_int32(label_cls), depth=self.num_classes + 1,
                on_value=1.0,
                off_value=0.0)
        
        with tf.variable_scope('segmentation_accuracy'):
            num_fg_pts = tf.reduce_sum(tf.cast(self._fg_mask, tf.float32)) / self._batch_size
            tf.summary.scalar('foreground_points_num', num_fg_pts)
            # seg accuracy
            all_ones = tf.ones_like(seg_preds, dtype=tf.float32)
            num_total_pts = tf.reduce_sum(all_ones)
            seg_correct = tf.equal(seg_preds, tf.to_int32(label_cls))
            #seg_accuracy = tf.reduce_mean(seg_correct)
            seg_accuracy = tf.reduce_sum(tf.cast(seg_correct, tf.float32)) / num_total_pts
            tf.summary.scalar('segmentation_accuracy', seg_accuracy)
        
        # Ground Truth Box Cls/Reg
        with tf.variable_scope('box_cls_reg_gt'):
            (bin_x_gt, res_x_gt, bin_z_gt, res_z_gt, 
             bin_theta_gt, res_theta_gt, res_y_gt, res_size_gt) = bin_based_box3d_encoder.tf_encode(
                fg_pts, 0, fg_label_boxes_3d, mean_sizes,
                self.S, self.DELTA, self.R, self.DELTA_THETA)
            
            bin_x_gt_one_hot = tf.one_hot(
                tf.to_int32(bin_x_gt), depth=int(2 * self.S / self.DELTA),
                on_value=1.0,
                off_value=0.0)
            
            bin_z_gt_one_hot = tf.one_hot(
                tf.to_int32(bin_z_gt), depth=int(2 * self.S / self.DELTA),
                on_value=1.0,
                off_value=0.0)
            
            bin_theta_gt_one_hot = tf.one_hot(
                tf.to_int32(bin_theta_gt), depth=int(2 * self.R / self.DELTA_THETA),
                on_value=1.0,
                off_value=0.0)
        
        ######################################################
        # Prediction Dict
        ######################################################
        predictions = dict()
        if self._train_val_test in ['train', 'val']:
            predictions[self.PRED_SEG_SOFTMAX] = seg_softmax
            predictions[self.PRED_SEG_GT] = segs_gt_one_hot
            predictions[self.PRED_TOTAL_PTS] = num_total_pts

            # Foreground BOX predictions
            predictions[self.PRED_FG_CLS] = (bin_x_logits, bin_z_logits, bin_theta_logits)
            predictions[self.PRED_FG_REG] = (res_x_norm, res_z_norm, res_theta_norm, res_y, res_size)

            # Foreground BOX ground truth
            predictions[self.PRED_FG_CLS_GT] = (bin_x_gt_one_hot, bin_z_gt_one_hot, bin_theta_gt_one_hot)
            predictions[self.PRED_FG_REG_GT] = (res_x_gt, res_z_gt, res_theta_gt, res_y_gt, res_size_gt)

            # Proposals after nms
            predictions[self.PRED_NMS_INDICES] = nms_indices
            predictions[self.PRED_PROPOSALS] = proposals
            predictions[self.PRED_OBJECTNESS_SOFTMAX] = fg_scores
        else:
            # self._train_val_test == 'test'
            predictions[self.PRED_SEG_SOFTMAX] = seg_softmax
            predictions[self.PRED_NMS_INDICES] = nms_indices
            predictions[self.PRED_PROPOSALS] = proposals
            predictions[self.PRED_OBJECTNESS_SOFTMAX] = fg_scores
        return predictions

    def _parse_rpn_output(self, rpn_output):
        '''
        Input:
            rpn_output: (B, p, NUM_BIN_X*2 + NUM_BIN_Z*2 + NUM_BIN_THETA*2 + 4
        Output:
            bin_x_logits
            res_x_norms
            
            bin_z_logits
            res_z_norms
            
            bin_theta_logits
            res_theta_norms
            
            res_y

            res_size: (l,w,h)
        '''
        bin_x_logits = tf.slice(rpn_output, [0, 0, 0], [-1, -1, self.NUM_BIN_X])
        res_x_norms = tf.slice(rpn_output, [0, 0, self.NUM_BIN_X], [-1, -1, self.NUM_BIN_X])
        
        bin_z_logits = tf.slice(rpn_output, [0, 0, self.NUM_BIN_X*2], [-1, -1, self.NUM_BIN_Z])
        res_z_norms = tf.slice(rpn_output, [0, 0, self.NUM_BIN_X*2+self.NUM_BIN_Z], [-1, -1, self.NUM_BIN_Z])

        bin_theta_logits = tf.slice(rpn_output, [0, 0, self.NUM_BIN_X*2+self.NUM_BIN_Z*2], [-1, -1, self.NUM_BIN_THETA])
        res_theta_norms = tf.slice(rpn_output, [0, 0, self.NUM_BIN_X*2+self.NUM_BIN_Z*2+self.NUM_BIN_THETA], 
                                              [-1, -1, self.NUM_BIN_THETA])

        res_y = tf.slice(rpn_output, [0, 0, self.NUM_BIN_X*2+self.NUM_BIN_Z*2+self.NUM_BIN_THETA*2], [-1, -1, 1])
        
        res_size = tf.slice(rpn_output, [0, 0, self.NUM_BIN_X*2+self.NUM_BIN_Z*2+self.NUM_BIN_THETA*2+1], [-1, -1, 3])

        return bin_x_logits, res_x_norms, bin_z_logits, res_z_norms, bin_theta_logits, res_theta_norms, res_y, res_size

    def create_feed_dict(self, batch_size=2, sample_index=None):
        """ Fills in the placeholders with the actual input values.
            Currently, only a batch size of 1 is supported

        Args:
            sample_index: optional, only used when train_val_test == 'test',
                a particular sample index in the dataset
                sample list to build the feed_dict for

        Returns:
            a feed_dict dictionary that can be used in a tensorflow session
        """
        if batch_size != self._batch_size :
            raise ValueError('feed batch_size must equal to model build batch_size')

        if self._train_val_test in ["train", "val"]:

            # sample_index should be None
            if sample_index is not None:
                raise ValueError('sample_index should be None. Do not load '
                                 'particular samples during train or val')

            if self._train_val_test == "train":
                # Get the a random sample from the remaining epoch
                samples = self.dataset.next_batch(batch_size=batch_size)

            else:  # self._train_val_test == "val"
                # Load samples in order for validation
                samples = self.dataset.next_batch(batch_size=batch_size,
                                                  shuffle=False)
        else:
            # For testing, any sample should work
            if sample_index is not None:
                samples = self.dataset.load_samples([sample_index])
            else:
                samples = self.dataset.next_batch(batch_size=batch_size, shuffle=False)
        
        if self._is_training:
            offset = int(random.gauss(0, self._pc_sample_pts * self._pc_sample_pts_variance))
            offset = max(offset, -self._pc_sample_pts * self._pc_sample_pts_clip)
            offset = min(offset, self._pc_sample_pts * self._pc_sample_pts_clip)
            pc_sample_pts = int(self._pc_sample_pts + offset)
        else:
            pc_sample_pts = self._pc_sample_pts
        
        batch_pc_inputs = []
        batch_label_segs = []
        batch_img_inputs = []
        batch_calib_inputs = []
        self._samples_info.clear()
        for sample in samples:
            # Network input data
            pc_input = sample.get(constants.KEY_POINT_CLOUD)
            label_seg = sample.get(constants.KEY_LABEL_SEG)
            img_input = sample.get(constants.KEY_IMAGE_INPUT)
            stereo_calib_p2 = sample.get(constants.KEY_STEREO_CALIB_P2)
            sample_name = sample.get(constants.KEY_SAMPLE_NAME)
            
            def random_sample(pool_size, sample_num):
                if pool_size > sample_num:
                    choices = np.random.choice(pool_size, sample_num, replace=False)
                else:
                    choices = np.concatenate((np.random.choice(pool_size, pool_size, replace=False),
                                              np.random.choice(pool_size, sample_num - pool_size, replace=True)))
                return choices
            
            pool_size = pc_input.shape[0]
            while True: 
                choices = random_sample(pool_size, pc_sample_pts)
                pc_input_sampled = pc_input[choices]
                label_seg_sampled = label_seg[choices]
                
                foreground_point_num = label_seg_sampled[label_seg_sampled[:, 0] > 0].shape[0]
                if foreground_point_num > 0 or self._train_val_test == 'test':
                    break
            
            batch_pc_inputs.append(pc_input_sampled)
            batch_label_segs.append(label_seg_sampled)
            img_input_resized = cv2.resize(img_input, (self._img_w, self._img_h))
            batch_img_inputs.append(img_input_resized)
            stereo_calib_p2[0,:] *= (self._img_w / img_input.shape[1])
            stereo_calib_p2[1,:] *= (self._img_h / img_input.shape[0])
            batch_calib_inputs.append(stereo_calib_p2)
            
            # Temporary sample info for debugging
            self._samples_info.append(sample_name)

        # this is a list to match the explicit shape for the placeholder
        # self._placeholder_inputs[self.PL_IMG_IDX] = [int(sample_name)]

        # Fill in the rest
        self._placeholder_inputs[self.PL_PC_INPUT] = np.asarray(batch_pc_inputs)
        self._placeholder_inputs[self.PL_LABEL_SEGS] = np.asarray(batch_label_segs)
        self._placeholder_inputs[self.PL_IMG_INPUT] = np.asarray(batch_img_inputs)
        self._placeholder_inputs[self.PL_CALIB_P2] = np.asarray(batch_calib_inputs)
        
        # Create a feed_dict and fill it with input values
        feed_dict = dict()
        for key, value in self.placeholders.items():
            feed_dict[value] = self._placeholder_inputs[key]

        return feed_dict

    def loss(self, prediction_dict):

        seg_softmax = prediction_dict[self.PRED_SEG_SOFTMAX]
        seg_gt = prediction_dict[self.PRED_SEG_GT]

        with tf.variable_scope('rpn_losses'):
            with tf.variable_scope('segmentation'):
                seg_loss = losses.WeightedFocalLoss()
                seg_loss_weight = self._config.loss_config.seg_loss_weight
                segmentation_loss = seg_loss(
                            seg_softmax,
                            seg_gt,
                            weight=seg_loss_weight)
                with tf.variable_scope('seg_norm'):
                    num_total_pts = prediction_dict[self.PRED_TOTAL_PTS]
                    segmentation_loss /= num_total_pts
                    tf.summary.scalar('segmentation', segmentation_loss)
            
            # these should include foreground pts only
            with tf.variable_scope('bin_classification'):
                cls_loss = losses.WeightedSoftmaxLoss()
                cls_loss_weight = self._config.loss_config.cls_loss_weight
                bin_classification_loss = 0.0
                #bin_x_logits, bin_z_logits, bin_theta_logits = prediction_dict[self.PRED_FG_CLS]
                #bin_x_gt_one_hot, bin_z_gt_one_hot, bin_theta_gt_one_hot = prediction_dict[self.PRED_FG_CLS_GT]
                for elem in zip(prediction_dict[self.PRED_FG_CLS], prediction_dict[self.PRED_FG_CLS_GT]):
                    bin_classification_loss += cls_loss(elem[0], elem[1],
                                                    weight=cls_loss_weight)
                with tf.variable_scope('cls_norm'):
                    # normalize by the number of foreground pts
                    num_fg_pts = self._batch_size * self.NUM_FG_POINT
                    bin_classification_loss /= num_fg_pts
                    tf.summary.scalar('bin_classification', bin_classification_loss)

            with tf.variable_scope('regression'):
                reg_loss = losses.WeightedSmoothL1Loss()
                reg_loss_weight = self._config.loss_config.reg_loss_weight
                regression_loss = 0.0
                #res_x_norm, res_z_norm, res_theta_norm, res_y, res_size = prediction_dict[self.PRED_FG_REG]
                #res_x_gt, res_z_gt, res_theta_gt, res_y_gt, res_size_gt = prediction_dict[self.PRED_FG_REG_GT]
                for elem in zip(prediction_dict[self.PRED_FG_REG], prediction_dict[self.PRED_FG_REG_GT]):
                    regression_loss += reg_loss(elem[0], elem[1], 
                                                weight=reg_loss_weight)
                with tf.variable_scope('reg_norm'):
                    # normalize by the number of foreground pts
                    regression_loss /= num_fg_pts
                    tf.summary.scalar('regression', regression_loss)
            
            with tf.variable_scope('rpn_loss'):
                rpn_loss = segmentation_loss + bin_classification_loss + regression_loss

        loss_dict = {
            self.LOSS_RPN_SEGMENTATION: segmentation_loss,
            self.LOSS_RPN_BIN_CLASSIFICATION: bin_classification_loss,
            self.LOSS_RPN_REGRESSION: regression_loss,
        }

        return loss_dict, rpn_loss
    
    def create_path_drop_masks(self,
                               p_img,
                               p_bev,
                               random_values):
        """Determines global path drop decision based on given probabilities.

        Args:
            p_img: A tensor of float32, probability of keeping image branch
            p_bev: A tensor of float32, probability of keeping bev branch
            random_values: A tensor of float32 of shape [3], the results
                of coin flips, values should range from 0.0 - 1.0.

        Returns:
            final_img_mask: A constant tensor mask containing either one or zero
                depending on the final coin flip probability.
            final_bev_mask: A constant tensor mask containing either one or zero
                depending on the final coin flip probability.
        """

        def keep_branch(): return tf.constant(1.0)

        def kill_branch(): return tf.constant(0.0)

        # The logic works as follows:
        # We have flipped 3 coins, first determines the chance of keeping
        # the image branch, second determines keeping bev branch, the third
        # makes the final decision in the case where both branches were killed
        # off, otherwise the initial img and bev chances are kept.

        img_chances = tf.case([(tf.less(random_values[0], p_img),
                                keep_branch)], default=kill_branch)

        bev_chances = tf.case([(tf.less(random_values[1], p_bev),
                                keep_branch)], default=kill_branch)

        # Decision to determine whether both branches were killed off
        third_flip = tf.logical_or(tf.cast(img_chances, dtype=tf.bool),
                                   tf.cast(bev_chances, dtype=tf.bool))
        third_flip = tf.cast(third_flip, dtype=tf.float32)

        # Make a second choice, for the third case
        # Here we use a 50/50 chance to keep either image or bev
        # If its greater than 0.5, keep the image
        img_second_flip = tf.case([(tf.greater(random_values[2], 0.5),
                                    keep_branch)],
                                  default=kill_branch)
        # If its less than or equal to 0.5, keep bev
        bev_second_flip = tf.case([(tf.less_equal(random_values[2], 0.5),
                                    keep_branch)],
                                  default=kill_branch)

        # Use lambda since this returns another condition and it needs to
        # be callable
        final_img_mask = tf.case([(tf.equal(third_flip, 1),
                                   lambda: img_chances)],
                                 default=lambda: img_second_flip)

        final_bev_mask = tf.case([(tf.equal(third_flip, 1),
                                   lambda: bev_chances)],
                                 default=lambda: bev_second_flip)

        return final_img_mask, final_bev_mask
