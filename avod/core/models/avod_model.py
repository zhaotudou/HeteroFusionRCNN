import numpy as np

import tensorflow as tf

from avod.builders import feature_extractor_builder
from avod.builders import avod_fc_layers_builder
from avod.builders import avod_loss_builder
from avod.core import anchor_projector
from avod.core import anchor_encoder
from avod.core import box_3d_encoder
from avod.core import box_8c_encoder
from avod.core import box_4c_encoder
from avod.core import pointfly as pf

from avod.core import box_list
from avod.core import box_list_ops

from avod.core import model
from avod.core import orientation_encoder
from avod.core.models.rpn_model import RpnModel


class AvodModel(model.DetectionModel):
    ##############################
    # Keys for Placeholders
    ##############################
    PL_PROPOSALS = 'proposals_pl'
    
    ##############################
    # Keys for Predictions
    ##############################
    # Mini batch (mb) cls
    PRED_MB_CLASSIFICATION_LOGITS = 'avod_mb_classification_logits'
    PRED_MB_CLASSIFICATIONS_GT = 'avod_mb_classifications_gt'
    PRED_MB_CLASSIFICATION_NUM = 'avod_mb_classification_num_boxes'

    # Mini batch (mb) cls-reg
    PRED_MB_CLS = 'avod_mb_cls'
    PRED_MB_REG = 'avod_mb_reg'
    PRED_MB_CLS_GT = 'avod_mb_cls_gt'
    PRED_MB_REG_GT = 'avod_mb_reg_gt'
    PRED_MB_POS_REG_MASK = 'avod_mb_pos_reg_mask'
    
    # Top predictions after BEV NMS
    PRED_TOP_CLASSIFICATION_LOGITS = 'avod_top_classification_logits'
    PRED_TOP_CLASSIFICATION_SOFTMAX = 'avod_top_classification_softmax'
    PRED_TOP_PREDICTION_BOXES_3D = 'avod_top_prediction_boxes_3d'

    PRED_MAX_IOUS = 'avod_max_ious'
    PRED_ALL_IOUS = 'avod_proposal_ious'

    ##############################
    # Keys for Loss
    ##############################
    LOSS_FINAL_CLASSIFICATION = 'avod_classification_loss'
    LOSS_FINAL_REGRESSION = 'avod_regression_loss'

    # (for debugging)
    LOSS_FINAL_ORIENTATION = 'avod_orientation_loss'
    LOSS_FINAL_LOCALIZATION = 'avod_localization_loss'

    def __init__(self, model_config, train_val_test, dataset, batch_size=1):
        """
        Args:
            model_config: configuration for the model
            train_val_test: "train", "val", or "test"
            dataset: the dataset that will provide samples and ground truth
        """

        # Sets model configs (_config)
        super(AvodModel, self).__init__(model_config)

        self._batch_size = batch_size
        # TODO: batch_size must be 1 
        if self._batch_size != 1:
            raise ValueError('Invalid batch_size, should be 1')
        self.dataset = dataset

        # Dataset config
        self.num_classes = dataset.num_classes

        # Input config
        input_config = self._config.input_config

        #self._img_pixel_size = np.asarray([input_config.img_dims_h,
        #                                   input_config.img_dims_w])
        #self._img_depth = [input_config.img_depth]

        # AVOD config
        avod_config = self._config.avod_config
        self._proposal_roi_crop_size = avod_config.avod_proposal_roi_crop_size
        self._positive_selection = avod_config.avod_positive_selection
        self._nms_size = avod_config.avod_nms_size
        self._nms_iou_threshold = avod_config.avod_nms_iou_thresh
        self._path_drop_probabilities = self._config.path_drop_probabilities
        self._box_rep = avod_config.avod_box_representation

        self.S = avod_config.avod_xz_search_range
        self.DELTA = avod_config.avod_xz_bin_len
        self.NUM_BIN_X = int(2 * self.S / self.DELTA)
        self.NUM_BIN_Z = self.NUM_BIN_X
        
        self.R = avod_config.avod_theta_search_range * np.pi
        self.DELTA_THETA = avod_config.avod_theta_bin_len * np.pi / 180
        self.NUM_BIN_THETA = int(2 * self.R / self.DELTA_THETA)

        self._pooling_context_length = avod_config.avod_pooling_context_length

        mini_batch_config = avod_config.mini_batch_config
        cls_iou_thresholds = mini_batch_config.cls_iou_3d_thresholds
        reg_iou_thresholds = mini_batch_config.reg_iou_3d_thresholds
        self.cls_neg_iou_range = [cls_iou_thresholds.neg_iou_lo, 
                                  cls_iou_thresholds.neg_iou_hi]
        self.cls_pos_iou_range = [cls_iou_thresholds.pos_iou_lo, 
                                  cls_iou_thresholds.pos_iou_hi]
        self.reg_neg_iou_range = [reg_iou_thresholds.neg_iou_lo, 
                                  reg_iou_thresholds.neg_iou_hi]
        self.reg_pos_iou_range = [reg_iou_thresholds.pos_iou_lo, 
                                  reg_iou_thresholds.pos_iou_hi]

        # Feature Extractor Nets
        self._pc_feature_extractor = \
            feature_extractor_builder.get_extractor(
                self._config.layers_config.avod_config.pc_feature_extractor)
        if self._box_rep not in ['box_3d', 'box_8c', 'box_8co',
                                 'box_4c', 'box_4ca']:
            raise ValueError('Invalid box representation', self._box_rep)

        # Create the RpnModel
        self._rpn_model = RpnModel(model_config, train_val_test, dataset, batch_size)

        if train_val_test not in ["train", "val", "test"]:
            raise ValueError('Invalid train_val_test value,'
                             'should be one of ["train", "val", "test"]')
        self._train_val_test = train_val_test
        self._is_training = (self._train_val_test == 'train')
        self.dataset.train_val_test = self._train_val_test

        # Network input placeholders
        self.placeholders = dict()

        # Inputs to network placeholders
        self._placeholder_inputs = dict()
        
    def _add_placeholder(self, dtype, shape, name):
        placeholder = tf.placeholder(dtype, shape, name)
        self.placeholders[name] = placeholder
        return placeholder

    def _set_up_input_pls(self):
        """Sets up input placeholders by adding them to self._placeholders.
        Keys are defined as self.PL_*.
        """
        with tf.variable_scope('pl_proposals'):
             self._proposals_pl = self._add_placeholder(tf.float32, [None, None, 7], self.PL_PROPOSALS)
    
    def _canonical_transform(self, pts, boxes_3d):
        '''
        Canonical Coordinate Transform
        Input:
            pts: (N,R,3) [x,y,z] float32
            boxes_3d:(N,7)  [cx,cy,cz,l,w,h,ry] float 32
        Output:
            pts_ct: (N,R,3) [x',y',z'] float32
        '''
        all_rys = boxes_3d[:, 6] * -1
        ry_sin = tf.sin(all_rys)
        ry_cos = tf.cos(all_rys)

        zeros = tf.zeros_like(all_rys, dtype=tf.float32)
        ones = tf.ones_like(all_rys, dtype=tf.float32)

        # Rotation matrix
        rot_mats = tf.stack([tf.stack([ry_cos, zeros, ry_sin], axis=1),
                             tf.stack([zeros, ones, zeros], axis=1),
                             tf.stack([-ry_sin, zeros, ry_cos], axis=1)],
                            axis=2)
        pts_rot = tf.matmul(rot_mats, pts, transpose_a=True, transpose_b=True)
        
        pts_ct_x = pts_rot[:,0] - tf.reshape(boxes_3d[:,0], (-1,1))
        pts_ct_y = pts_rot[:,1] - tf.reshape(boxes_3d[:,1], (-1,1))
        pts_ct_z = pts_rot[:,2] - tf.reshape(boxes_3d[:,2], (-1,1))

        pts_ct = tf.stack([pts_ct_x, pts_ct_y, pts_ct_z], axis=1)

        return tf.matrix_transpose(pts_ct)

    def _gather_residuals(self, res_x_norms, res_z_norms, res_theta_norms,
                                bin_x, bin_z, bin_theta):
        
        '''
        Input:
            res_x_norms: (N,K)
            bin_x:(N)
        return:
            res_x_norm: (N)
        '''

        #TF version: (if N is not None)
        ##########
        N = bin_x.shape[0].value
        Ns = tf.reshape(tf.range(N), (N,1))

        NK_x = tf.concat([Ns, tf.reshape(bin_x, (N,1))], axis=1) # (N,2)
        res_x_norm = tf.gather_nd(res_x_norms, NK_x) #(N)
        
        NK_z = tf.concat([Ns, tf.reshape(bin_z, (N,1))], axis=1) # (N,2)
        res_z_norm = tf.gather_nd(res_z_norms, NK_z) #(N)
        
        NK_theta = tf.concat([Ns, tf.reshape(bin_theta, (N,1))], axis=1) # (N,2)
        res_theta_norm = tf.gather_nd(res_theta_norms, NK_theta) #(N)
    
        '''
        NumPy version: if N is None, by using tf.py_func, p should be determined
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
            cls_preds: (N), [klass], kclass is 1-based, 0-background
        Output
            mean_sizes: (N,3) [l,w,h]
        '''

        #TF version: (if p is not None)
        ##########
        N = cls_preds.shape[0].value
        
        Ns = tf.reshape(tf.range(N), (N,1))

        K_mean_sizes = tf.reshape(cluster_sizes, (-1,3))
        NK_mean_sizes = tf.tile(tf.expand_dims(K_mean_sizes, 0), [N,1,1])

        NK = tf.concat([Ns, tf.reshape(cls_preds, (N,1))], axis=1) # (N,2)
        
        mean_sizes = tf.gather_nd(NK_mean_sizes, NK)

        '''
        NumPy version: if p is None, by using tf.py_func, p should be determined
        #############
        K_mean_sizes = np.reshape(cluster_sizes, (-1,3))
        K_mean_sizes = np.vstack([np.asarray([0.0, 0.0, 0.0]), K_mean_sizes]) # insert 0-background
        mean_sizes = K_mean_sizes[cls_preds]
        
        return mean_sizes.astype(np.float32)
        '''
        
        return mean_sizes
    
    def build(self):
        rpn_model = self._rpn_model

        # Share the same prediction dict as RPN
        prediction_dict = rpn_model.build()
        
        if self.model_config.alternating_training_step == 2:
            self._set_up_input_pls()
            top_proposals = self.placeholders[AvodModel.PL_PROPOSALS]   #(B,n,7)
        else:
            top_proposals = prediction_dict[RpnModel.PRED_TOP_PROPOSALS]#(B,n,7)

        # Expand proposals' size
        with tf.variable_scope('expand_proposal'):
            expand_length = self._pooling_context_length
            expanded_size = top_proposals[:,:,3:6] + expand_length
            expanded_proposals = tf.stack([
                top_proposals[:,:,0],
                top_proposals[:,:,1],
                top_proposals[:,:,2],
                expanded_size[:,:,0],
                expanded_size[:,:,1],
                expanded_size[:,:,2],
                top_proposals[:,:,6],
            ], axis=2)  #(B,n,7)
            
        pc_pts = rpn_model._pc_pts  #(B,P,3)
        pc_fts = rpn_model._pc_fts  #(B,P,C)
        foreground_mask = prediction_dict[RpnModel.PRED_FG_MASK] #(B,P)
        '''
        if not (self._path_drop_probabilities[0] ==
                self._path_drop_probabilities[1] == 1.0):

            with tf.variable_scope('avod_path_drop'):

                img_mask = rpn_model.img_path_drop_mask
                bev_mask = rpn_model.bev_path_drop_mask

                img_feature_maps = tf.multiply(img_feature_maps,
                                               img_mask)

                bev_feature_maps = tf.multiply(bev_feature_maps,
                                               bev_mask)
        else:
            bev_mask = tf.constant(1.0)
            img_mask = tf.constant(1.0)
        '''
        # ROI Pooling
        with tf.variable_scope('avod_roi_pooling'):
            def get_box_indices(boxes):
                proposals_shape = boxes.get_shape().as_list()
                if any(dim is None for dim in proposals_shape):
                    proposals_shape = tf.shape(boxes)
                ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
                multiplier = tf.expand_dims(
                    tf.range(start=0, limit=proposals_shape[0]), 1)
                return tf.reshape(ones_mat * multiplier, [-1])

            # These should be all 0's since there is only 1 image
            tf_box_indices = get_box_indices(expanded_proposals)

            # Do ROI Pooling on PC
            from cropping import tf_cropping
            top_proposals = tf.reshape(top_proposals, (-1,7)) #(N=Bn,7)
            expanded_proposals = tf.reshape(expanded_proposals, (-1,7)) #(N=Bn,7)
            crop_pts, crop_fts, crop_mask, _, non_empty_box_mask = tf_cropping.pc_crop_and_sample(
                pc_pts,
                pc_fts,
                foreground_mask,
                box_8c_encoder.tf_box_3d_to_box_8co(expanded_proposals),
                tf_box_indices,
                self._proposal_roi_crop_size)   #(N,R,3), (N,R,C), (N,R), _, (N)

            crop_pts = tf.boolean_mask(crop_pts, non_empty_box_mask)    #(N',R,3)
            crop_fts = tf.boolean_mask(crop_fts, non_empty_box_mask)
            crop_mask = tf.boolean_mask(crop_mask, non_empty_box_mask)
            top_proposals = tf.boolean_mask(top_proposals, non_empty_box_mask)
            expanded_proposals = tf.boolean_mask(expanded_proposals, non_empty_box_mask)
            tf.summary.histogram('non_empty_box_mask', tf.cast(non_empty_box_mask, tf.int8))

            '''
            # Do ROI Pooling on image
            img_rois = tf.image.crop_and_resize(
                img_feature_maps,
                img_proposal_boxes_norm_tf_order,
                tf_box_indices,
                self._proposal_roi_crop_size,
                name='img_rois')
            '''
        with tf.variable_scope('local_spatial_feature'):
            with tf.variable_scope('canonical_transform'):
                crop_pts_ct = self._canonical_transform(crop_pts, expanded_proposals)

            with tf.variable_scope('distance_to_sensor'):
                crop_distance = tf.sqrt(
                                    tf.square(crop_pts[:,:,0]) + \
                                    tf.square(crop_pts[:,:,1]) + \
                                    tf.square(crop_pts[:,:,2])
                                )

            local_feature_input = tf.concat([crop_pts_ct, 
                                            tf.expand_dims(tf.to_float(crop_mask), -1), 
                                            tf.expand_dims(crop_distance, -1)], 
                                           axis=-1)
            
            with tf.variable_scope('mlp'):
                fc_layers = [local_feature_input]
                layers_config = self._config.layers_config.avod_config.mlp
                for layer_idx, layer_param in enumerate(layers_config):
                    print(layer_param)
                    C = layer_param.C
                    dropout_rate = layer_param.dropout_rate
                    fc = pf.dense(fc_layers[-1], C, 'fc{:d}'.format(layer_idx), self._is_training)
                    fc_drop = tf.layers.dropout(fc, dropout_rate, training=self._is_training, 
                                                name='fc{:d}_drop'.format(layer_idx))
                    fc_layers.append(fc_drop)
                
                fc_output = pf.dense(fc_layers[-1], crop_fts.shape[2].value, 
                                     'fc_output', self._is_training, activation=None) #(N',R,C)

        with tf.variable_scope('pc_encoder'):
            merged_fts = tf.concat([crop_fts, fc_output], axis=-1)   #(N',R,2C)
            encode_pts, encode_fts = self._pc_feature_extractor.build(
                                        crop_pts, merged_fts, self._is_training) # (N',r,3), (N',r,C')
        
        #branch-1: Box classification 
        #########################################
        with tf.variable_scope('classification_confidence'):
            cls_logits = pf.dense(encode_fts, self.num_classes + 1, 'cls_logits', 
                                  self._is_training, with_bn=False, activation=None) #(N',r,K)
            cls_mean_logits = tf.reduce_mean(cls_logits, axis=1)    #(N',K)
            cls_softmax = tf.nn.softmax(cls_mean_logits, name='cls_softmax') 
            cls_preds = tf.argmax(cls_softmax, axis=-1, name='cls_predictions')
            cls_scores = tf.reduce_max(cls_softmax, axis=-1, name='cls_scores')


        #branch-2: bin-based 3D box refinement
        #########################################
        with tf.variable_scope('bin_based_box_refinement'):
            # Parse brn layers config
            encode_mean_fts = tf.reduce_mean(encode_fts, axis=1)   #(N',C')
            fc_layers = [encode_mean_fts]
            layers_config = self._config.layers_config.avod_config.fc_layer
            for layer_idx, layer_param in enumerate(layers_config):
                print(layer_param)
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
            res_y, res_size = self._parse_brn_output(fc_output)
            res_y = tf.squeeze(res_y, [-1])

        # Final Predictions
        ######################################################
        with tf.variable_scope('boxes'):
            # Decode bin-based 3D Box 
            bin_x = tf.argmax(bin_x_logits, axis=-1) #(N')
            bin_z = tf.argmax(bin_z_logits, axis=-1) #(N')
            bin_theta = tf.argmax(bin_theta_logits, axis=-1) #(N')
            
            res_x_norm, res_z_norm, res_theta_norm = self._gather_residuals(
                res_x_norms, res_z_norms, res_theta_norms, bin_x, bin_z, bin_theta)
            
            mean_sizes = self._gather_mean_sizes(
                tf.convert_to_tensor(np.asarray(self._cluster_sizes)), cls_preds)

            with tf.variable_scope('decoding'):
                reg_boxes_3d = bin_based_box3d_encoder.tf_decode(
                        expanded_proposals[:,:3], expanded_proposals[:,6],
                        bin_x, res_x_norm,
                        bin_z, res_z_norm,
                        bin_theta, res_theta_norm,
                        res_y, res_size, mean_sizes,
                        self.S, self.DELTA, self.R, self.DELTA_THETA) # (N',7)
            
            oriented_NMS = False
            print("oriented_NMS = " + str(oriented_NMS))
            # BEV projection
            with tf.variable_scope('bev_projection'):
                if oriented_NMS: 
                    bev_boxes, _ = tf.py_func(
                        box_3d_projector.project_to_bev,
                        [reg_boxes_3d, tf.constant(self._bev_extents)],
                        (tf.float32, tf.float32))
            
                else:
                    # ortho rotating
                    box_anchors = box_3d_encoder.tf_box_3d_to_anchor(reg_boxes_3d)
                    bev_boxes, _ = anchor_projector.project_to_bev(box_anchors, self._bev_extents)
                    bev_boxes_tf_order = anchor_projector.reorder_projected_boxes(bev_boxes)

            # bev-NMS and ignore multiclass
            with tf.variable_scope('bev_nms'):
                if oriented_NMS: 
                    nms_masks = tf.py_func(
                        oriented_nms.nms,
                        [bev_boxes, cls_scores, 
                        tf.constant(self._nms_iou_thresh), tf.constant(self._nms_size)],
                        tf.bool)
                
                    top_boxes_3d = tf.reshape(
                                        tf.boolean_mask(reg_boxes_3d, nms_masks),
                                        [-1, 7])
                    top_cls_logits = tf.boolean_mask(cls_mean_logits, nms_masks)
                    top_cls_softmax = tf.boolean_mask(cls_softmax, nms_masks)
                    top_cls_scores = tf.boolean_mask(cls_scores, nms_masks)
                else:
                    nms_indices = tf.image.non_max_suppression(
                        bev_boxes_tf_order,
                        cls_scores,
                        max_output_size=self._nms_size,
                        iou_threshold=self._nms_iou_thresh)
                    
                    top_boxes_3d = tf.gather(reg_boxes_3d, nms_indices)
                    top_cls_logits = tf.gather(cls_mean_logits, nms_indices)
                    top_cls_softmax = tf.gather(cls_softmax, nms_indices)
                    top_cls_scores = tf.gather(cls_scores, nms_indices)
                
                tf.summary.histogram('top_cls_scores', top_objectness_scores)
        
        ######################################################
        # Determine Positive/Negative GTs for the loss function & metrics
        ######################################################
        top_proposal_iou3ds = tf.boolean_mask(
                                tf.reshape(prediction_dict[RpnModel.PRED_TOP_PROPOSAL_IOU3DS], [-1]),
                                non_empty_box_mask) #(N') 
        top_proposal_gt_cls = tf.boolean_mask(
                                tf.reshape(prediction_dict[RpnModel.PRED_TOP_PROPOSAL_GT_CLS], [-1]),
                                non_empty_box_mask) #(N')
        top_proposal_gt_boxes = tf.boolean_mask(
                                tf.reshape(prediction_dict[RpnModel.PRED_TOP_PROPOSAL_GT_BOXES], [-1,7]),
                                non_empty_box_mask) #(N',7)
        
        # for box cls loss
        with tf.variable_scope('box_cls_gt'):
            neg_cls_mask = tf.less(proposal_iou3ds, self.cls_neg_iou_range[1])
            pos_cls_mask = tf.greater(proposal_iou3ds, self.cls_pos_iou_range[0])
            mb_cls_mask = tf.logical_or(neg_cls_mask, pos_cls_mask)
            mb_cls_num_boxes = tf.reduce_sum(tf.cast(mb_cls_mask, tf.float32))
            
            # cls preds
            print('cls_logits shape')
            neg_cls_logits = tf.boolean_mask(cls_mean_logits, neg_cls_mask)
            print(neg_cls_logits.shape)
            pos_cls_logits = tf.boolean_mask(cls_mean_logits, pos_cls_mask)
            print(pos_cls_logits.shape)
            mb_cls_logits = tf.concat([neg_cls_logits, pos_cls_logits], axis=0)
            print(mb_cls_logits.shape)
            
            # cls gt
            neg_cls_gt = tf.boolean_mask(top_proposal_gt_cls, neg_cls_mask)
            neg_cls_gt = tf.zeros_like(neg_cls_gt)
            pos_cls_gt = tf.boolean_mask(top_proposal_gt_cls, pos_cls_mask)
            mb_cls_gt = tf.concat([neg_cls_gt, pos_cls_gt], axis=0)
            mb_cls_gt_one_hot = tf.one_hot(
                            tf.to_int64(all_cls_gt), depth=self.num_classes + 1,
                            on_value=1.0,
                            off_value=0.0)
           
        # for box refinement loss
        with tf.variable_scope('box_cls_reg_gt'):
            pos_reg_mask = tf.greater(proposal_iou3ds, self.reg_pos_iou_range[0])
            #pos_reg_num_boxes = tf.reduce_sum(tf.cast(pos_reg_mask, tf.float32))

            # reg gt
            (bin_x_gt, res_x_gt, bin_z_gt, res_z_gt, 
             bin_theta_gt, res_theta_gt, res_y_gt, res_size_gt) = bin_based_box3d_encoder.tf_encode(
                top_proposals[:,:3], top_proposals[:,6], 
                top_proposal_gt_boxes, mean_sizes,
                self.S, self.DELTA, self.R, self.DELTA_THETA)
            
            bin_x_gt_one_hot = tf.one_hot(
                tf.to_int64(bin_x_gt), depth=int(2 * self.S / self.DELTA),
                on_value=1.0,
                off_value=0.0)
            
            bin_z_gt_one_hot = tf.one_hot(
                tf.to_int64(bin_z_gt), depth=int(2 * self.S / self.DELTA),
                on_value=1.0,
                off_value=0.0)
            
            bin_theta_gt_one_hot = tf.one_hot(
                tf.to_int64(bin_theta_gt), depth=int(2 * self.R / self.DELTA_THETA),
                on_value=1.0,
                off_value=0.0)


        ######################################################
        # Prediction Dict
        ######################################################
        if self._train_val_test in ['train', 'val']:
            # cls Mini batch preds & gt
            prediction_dict[self.PRED_MB_CLASSIFICATION_LOGITS] = mb_cls_logits
            prediction_dict[self.PRED_MB_CLASSIFICATIONS_GT] = mb_cls_gt_one_hot
            prediction_dict[self.PRED_MB_CLASSIFICATIONS_NUM] = mb_cls_num_boxes

            # reg Mini batch preds
            prediction_dict[self.PRED_MB_CLS] = (bin_x_loigts, bin_z_logits, bin_theta_logits)
            prediction_dict[self.PRED_MB_REG] = (res_x_norm, res_z_norm, res_theta_norm, res_y, res_size)

            # reg Mini batch gt
            prediction_dict[self.PRED_MB_CLS_GT] = (bin_x_gt_one_hot, bin_z_gt_one_hot, bin_theta_gt_one_hot)
            prediction_dict[self.PRED_MB_REG_GT] = (res_x_gt, res_z_gt, res_theta_gt, res_y_gt, res_size_gt)

            # reg Mini batch pos mask
            prediction_dict[self.PRED_MB_POS_REG_MASK] = pos_reg_mask
            
            # Top NMS predictions
            prediction_dict[self.PRED_TOP_CLASSIFICATION_LOGITS] = top_cls_logits
            prediction_dict[self.PRED_TOP_CLASSIFICATION_SOFTMAX] = top_cls_softmax
            prediction_dict[self.PRED_TOP_PREDICTION_BOXES_3D] = top_boxes_3d


        else:
            prediction_dict[self.PRED_TOP_CLASSIFICATION_LOGITS] = top_cls_logits
            prediction_dict[self.PRED_TOP_CLASSIFICATION_SOFTMAX] = top_cls_softmax
            prediction_dict[self.PRED_TOP_PREDICTION_BOXES_3D] = top_boxes_3d

        # prediction_dict[self.PRED_MAX_IOUS] = max_ious
        # prediction_dict[self.PRED_ALL_IOUS] = all_ious

        return prediction_dict

    def _parse_brn_output(self, brn_output):
        '''
        Input:
            brn_output: (N, NUM_BIN_X*2 + NUM_BIN_Z*2 + NUM_BIN_THETA*2 + 4
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
        bin_x_logits = tf.slice(brn_output, [0, 0], [-1, self.NUM_BIN_X])
        res_x_norms = tf.slice(brn_output,  [0, self.NUM_BIN_X], [-1, self.NUM_BIN_X])
        
        bin_z_logits = tf.slice(brn_output, [0, self.NUM_BIN_X*2], [-1, self.NUM_BIN_Z])
        res_z_norms = tf.slice(brn_output, [0, self.NUM_BIN_X*2+self.NUM_BIN_Z], [-1, self.NUM_BIN_Z])

        bin_theta_logits = tf.slice(brn_output, [0, self.NUM_BIN_X*2+self.NUM_BIN_Z*2], [-1, self.NUM_BIN_THETA])
        res_theta_norms = tf.slice(brn_output, [0, self.NUM_BIN_X*2+self.NUM_BIN_Z*2+self.NUM_BIN_THETA], 
                                               [-1, self.NUM_BIN_THETA])

        res_y = tf.slice(brn_output, [0, self.NUM_BIN_X*2+self.NUM_BIN_Z*2+self.NUM_BIN_THETA*2], [-1, 1])
        
        res_size = tf.slice(brn_output, [0, self.NUM_BIN_X*2+self.NUM_BIN_Z*2+self.NUM_BIN_THETA*2+1], [-1, 3])

        return bin_x_logits, res_x_norms, bin_z_logits, res_z_norms, bin_theta_logits, res_theta_norms, res_y, res_size
    
    def create_feed_dict(self, batch_size=1):
        feed_dict = self._rpn_model.create_feed_dict(batch_size)
        if self.model_config.alternating_training_step == 2:
            batch_proposals = []
            for sample_name in self._rpn_model.samples_info:
                proposal = self.dataset.load_proposals(sample_name)
                batch_proposals.append(proposal)
            self._placeholder_inputs[self.PL_PROPOSALS] = np.asarray(batch_proposals)
            
            for key, value in self.placeholders.items():
                feed_dict[value] = self._placeholder_inputs[key]
        return feed_dict

    def loss(self, prediction_dict):
        if self.model_config.alternating_training_step == 2:
            loss_dict = {}
            rpn_loss = tf.constant(0.0)
        else:
            loss_dict, rpn_loss = self._rpn_model.loss(prediction_dict)
        
        # cls Mini batch preds & gt
        mb_cls_logits = prediction_dict[self.PRED_MB_CLASSIFICATION_LOGITS]
        mb_cls_gt_one_hot = prediction_dict[self.PRED_MB_CLASSIFICATIONS_GT]

        
        with tf.variable_scope('brn_losses'):
            with tf.variable_scope('box_classification'):
                cls_loss = losses.WeightedSoftmaxLoss()
                cls_loss_weight = self._config.loss_config.cls_loss_weight
                box_classification_loss = cls_loss(mb_cls_logits, mb_cls_gt_one_hot, cls_loss_weight)
                
                with tf.variable_scope('cls_norm'):
                    # normalize by the number of boxes
                    mb_cls_num_boxes = prediction_dict[self.PRED_MB_CLASSIFICATIONS_NUM]
                    with tf.control_dependencies(
                        [tf.assert_positive(mb_cls_num_boxes)]):
                        box_classification_loss /= mb_cls_num_boxes
                    tf.summary.scalar('box_classification', box_classification_loss)

            # these should include positive boxes only
            with tf.variable_scope('bin_classification'):
                # reg Mini batch pos mask
                pos_reg_mask = prediction_dict[self.PRED_MB_POS_REG_MASK]
                bin_classification_loss = 0.0
                #bin_x_logits, bin_z_logits, bin_theta_logits = prediction_dict[self.PRED_MB_CLS]
                #bin_x_gt_one_hot, bin_z_gt_one_hot, bin_theta_gt_one_hot = prediction_dict[self.PRED_MB_CLS_GT]
                for elem in zip(prediction_dict[self.PRED_MB_CLS], prediction_dict[self.PRED_MB_CLS_GT]):
                    bin_classification_loss += cls_loss(elem[0], elem[1],
                                                    weight=cls_loss_weight,
                                                    mask=pos_reg_mask)
                with tf.variable_scope('cls_norm'):
                    # normalize by the number of positive boxes
                    num_pos_boxes = tf.reduce_sum(tf.cast(pos_reg_mask, tf.float32))
                    with tf.control_dependencies(
                        [tf.assert_positive(num_pos_boxes)]):
                        bin_classification_loss /= num_pos_boxes
                    tf.summary.scalar('bin_classification', bin_classification_loss)
            
            # these should include positive boxes only
            with tf.variable_scope('regression'):
                reg_loss = losses.WeightedSmoothL1Loss()
                reg_loss_weight = self._config.loss_config.reg_loss_weight
                regression_loss = 0.0
                #res_x_norm, res_z_norm, res_theta_norm, res_y, res_size = prediction_dict[self.PRED_MB_REG]
                #res_x_gt, res_z_gt, res_theta_gt, res_y_gt, res_size_gt = prediction_dict[self.PRED_MB_REG_GT]
                for elem in zip(prediction_dict[self.PRED_FG_REG], prediction_dict[self.PRED_FG_REG_GT]):
                    regression_loss += reg_loss(elem[0], elem[1], 
                                                weight=reg_loss_weight,
                                                mask=pos_reg_mask)
                with tf.variable_scope('reg_norm'):
                    # normalize by the number of positive boxes
                    with tf.control_dependencies(
                        [tf.assert_positive(num_pos_boxes)]):
                        regression_loss /= num_pos_boxes
                    tf.summary.scalar('regression', regression_loss)
            
            with tf.variable_scope('brn_loss'):
                brn_loss = box_classification_loss + bin_classification_loss + regression_loss
        
        loss_dict.update({self.LOSS_FINAL_CLASSIFICATION: box_classification_loss})
        loss_dict.update({self.LOSS_FINAL_BIN_CLASSIFICATION: bin_classification_loss})
        loss_dict.update({self.LOSS_FINAL_REGRESSION: regression_loss})

        with tf.variable_scope('total_loss'):
            total_loss = rpn_loss + brn_loss

        return loss_dict, total_loss
