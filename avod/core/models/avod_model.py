import numpy as np

import tensorflow as tf

from avod.builders import avod_fc_layers_builder
from avod.builders import avod_loss_builder
from avod.core import anchor_projector
from avod.core import anchor_encoder
from avod.core import box_3d_encoder
from avod.core import box_8c_encoder
from avod.core import box_4c_encoder

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
    # Mini batch (mb) ground truth
    PRED_MB_CLASSIFICATIONS_GT = 'avod_mb_classifications_gt'
    PRED_MB_OFFSETS_GT = 'avod_mb_offsets_gt'
    PRED_MB_ORIENTATIONS_GT = 'avod_mb_orientations_gt'

    # Mini batch (mb) predictions
    PRED_MB_CLASSIFICATION_LOGITS = 'avod_mb_classification_logits'
    PRED_MB_CLASSIFICATION_SOFTMAX = 'avod_mb_classification_softmax'
    PRED_MB_OFFSETS = 'avod_mb_offsets'
    PRED_MB_ANGLE_VECTORS = 'avod_mb_angle_vectors'

    # Top predictions after BEV NMS
    PRED_TOP_CLASSIFICATION_LOGITS = 'avod_top_classification_logits'
    PRED_TOP_CLASSIFICATION_SOFTMAX = 'avod_top_classification_softmax'

    PRED_TOP_PREDICTION_ANCHORS = 'avod_top_prediction_anchors'
    PRED_TOP_PREDICTION_BOXES_3D = 'avod_top_prediction_boxes_3d'
    PRED_TOP_ORIENTATIONS = 'avod_top_orientations'

    # Other box representations
    PRED_TOP_BOXES_8C = 'avod_top_regressed_boxes_8c'
    PRED_TOP_BOXES_4C = 'avod_top_prediction_boxes_4c'

    # Mini batch (mb) predictions (for debugging)
    PRED_MB_MASK = 'avod_mb_mask'
    PRED_MB_POS_MASK = 'avod_mb_pos_mask'
    PRED_MB_ANCHORS_GT = 'avod_mb_anchors_gt'
    PRED_MB_CLASS_INDICES_GT = 'avod_mb_gt_classes'

    # All predictions (for debugging)
    PRED_ALL_CLASSIFICATIONS = 'avod_classifications'
    PRED_ALL_OFFSETS = 'avod_offsets'
    PRED_ALL_ANGLE_VECTORS = 'avod_angle_vectors'

    PRED_MAX_IOUS = 'avod_max_ious'
    PRED_ALL_IOUS = 'avod_anchor_ious'

    ##############################
    # Keys for Loss
    ##############################
    LOSS_FINAL_CLASSIFICATION = 'avod_classification_loss'
    LOSS_FINAL_REGRESSION = 'avod_regression_loss'

    # (for debugging)
    LOSS_FINAL_ORIENTATION = 'avod_orientation_loss'
    LOSS_FINAL_LOCALIZATION = 'avod_localization_loss'

    def __init__(self, model_config, train_val_test, dataset):
        """
        Args:
            model_config: configuration for the model
            train_val_test: "train", "val", or "test"
            dataset: the dataset that will provide samples and ground truth
        """

        # Sets model configs (_config)
        super(AvodModel, self).__init__(model_config)

        self.dataset = dataset

        # Dataset config
        self._num_final_classes = self.dataset.num_classes + 1

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

        # Feature Extractor Nets
        self._pc_feature_extractor = \
            feature_extractor_builder.get_extractor(
                self._config.layers_config.avod_config.pc_feature_extractor)
        if self._box_rep not in ['box_3d', 'box_8c', 'box_8co',
                                 'box_4c', 'box_4ca']:
            raise ValueError('Invalid box representation', self._box_rep)

        # Create the RpnModel
        self._rpn_model = RpnModel(model_config, train_val_test, dataset)

        if train_val_test not in ["train", "val", "test"]:
            raise ValueError('Invalid train_val_test value,'
                             'should be one of ["train", "val", "test"]')
        self._train_val_test = train_val_test
        self._is_training = (self._train_val_test == 'train')
        self.dataset.train_val_test = self._train_val_test

        self.sample_info = {}
        
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
             self._proposals_pl = self._add_placeholder(tf.float32, [None, 6], self.PL_PROPOSALS)
    
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

    def build(self):
        rpn_model = self._rpn_model

        # Share the same prediction dict as RPN
        prediction_dict = rpn_model.build()
        
        if self.model_config.alternating_training_step == 2:
            self._set_up_input_pls()
            top_proposals = self.placeholders[AvodModel.PL_PROPOSALS]   #(B,n,7)
        else:
            top_proposals = prediction_dict[RpnModel.PRED_TOP_PROPOSALS]

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
        #img_feature_maps = rpn_model.img_feature_maps
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
            expanded_proposals = tf.reshape(expanded_proposals, (-1,7)) #(N=Bn,7)
            crop_pts, crop_fts, crop_mask, _, non_empty_box_mask = tf_cropping.pc_crop_and_sample(
                pc_pts,
                pc_fts,
                foreground_mask,
                box_8c_encoder.tf_box_3d_to_box_8co(expanded_proposals),
                tf_box_indices,
                self._proposal_roi_crop_size)   #(N,R,3), (N,R,C), (N,R), _, (N)

            non_empty_crop_pts = tf.boolean_mask(crop_pts, non_empty_box_mask)
            tf.summary.histogram('non_empty_box', tf.cast(non_empty_box, tf.int8))

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

            local_feature_input = tf.stack([crop_pts_ct, 
                                            tf.expand_dims(crop_mask, -1), 
                                            tf.expand_dims(crop_distance, -1)], 
                                           axis=2)
            
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
                                     'fc_output', self._is_training, activation=None) #(N,R,C)

        with tf.variable_scope('pc_encoder'):
            merged_fts = tf.stack([crop_fts, fc_output], axis=-1)   #(N,R,2C)
            encode_pts, encode_fts = self._pc_feature_extractor.build(
                                        crop_pts, merged_fts, self._is_training) # (N,r,3), (N,r,C')
        
        #branch-1: Box classification 
        #########################################
        with tf.variable_scope('classification_confidence'):
            cls_logits = pf.dense(encode_fts, self.num_classes + 1, 'cls_logits', 
                                  self._is_training, with_bn=False, activation=None) #(N,r,K)
            cls_mean_logits = tf.reduce_mean(cls_logits, axis=1)    #(N,K)
            cls_softmax = tf.nn.softmax(cls_mean_logits, name='cls_softmax') 
            #cls_preds = tf.argmax(cls_softmax, axis=-1, name='cls_predictions')
            cls_scores = tf.reduce_max(cls_softmax, axis=-1, name='cls_scores')


        #branch-2: bin-based 3D box refinement
        #########################################
        with tf.variable_scope('bin_based_box_refinement'):
            # Parse brn layers config
            encode_mean_fts = tf.reduce_mean(encode_fts, axis=1)   #(N,C')
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

        # Return final Boxes
        with tf.variable_scope('boxes'):
            #TODO
            # Decode bin-based 3D Box 
            bin_x = tf.argmax(bin_x_logits, axis=-1) #(B,p)
            bin_z = tf.argmax(bin_z_logits, axis=-1) #(B,p)
            bin_theta = tf.argmax(bin_theta_logits, axis=-1) #(B,p)
            
            res_x_norm, res_z_norm, res_theta_norm = tf.py_func(
                self._gather_residuals,
                [res_x_norms, res_z_norms, res_theta_norms, bin_x, bin_z, bin_theta], 
                (tf.float32, tf.float32, tf.float32))
            
            mean_sizes = tf.py_func(
                self._gather_mean_sizes,
                [tf.convert_to_tensor(np.asarray(self._cluster_sizes)), foreground_preds],
                tf.float32)

            with tf.variable_scope('decoding'):
                proposals = bin_based_box3d_encoder.tf_decode(
                        foreground_pts,
                        bin_x, res_x_norm,
                        bin_z, res_z_norm,
                        bin_theta, res_theta_norm,
                        res_y, res_size, mean_sizes,
                        self.S, self.DELTA, self.R, self.DELTA_THETA) # (B,p,7)
            
            oriented_NMS = False
            print("oriented_NMS = " + str(oriented_NMS))
            # BEV projection
            with tf.variable_scope('bev_projection'):
                if oriented_NMS: 
                    def single_batch_project_to_bev_fn(single_batch_proposals):
                        single_batch_bev_proposal_boxes, _ = tf.py_func(
                            box_3d_projector.project_to_bev,
                            [single_batch_proposals, tf.constant(self._bev_extents)],
                            (tf.float32, tf.float32))
                        return single_batch_bev_proposal_boxes
            
                    bev_proposal_boxes = tf.map_fn(
                        single_batch_project_to_bev_fn,
                        elems=proposals,
                        dtype=tf.float32)
                    bev_proposal_boxes = tf.reshape(bev_proposal_boxes, 
                                                    [self._batch_size, -1, 4, 2])
                else:
                    def single_batch_project_to_bev_fn(single_batch_proposals):
                        # ortho rotating
                        single_batch_proposal_anchors = box_3d_encoder.tf_box_3d_to_anchor(
                                                        single_batch_proposals)
                        single_batch_bev_proposal_boxes, _ = anchor_projector.project_to_bev(
                                                                single_batch_proposal_anchors, 
                                                                self._bev_extents)
                        single_batch_bev_proposal_boxes_tf_order = anchor_projector.reorder_projected_boxes(
                                single_batch_bev_proposal_boxes)

                        return single_batch_bev_proposal_boxes_tf_order

                    bev_proposal_boxes_tf_order = tf.map_fn(
                        single_batch_project_to_bev_fn,
                        elems=proposals,
                        dtype=tf.float32)
            
            # bev-NMS and ignore multiclass
            with tf.variable_scope('bev_nms'):
                if oriented_NMS: 
                    def single_batch_oriented_nms_fn(args):
                        (single_batch_boxes, single_batch_scores) = args
                        single_batch_nms_mask = tf.py_func(
                            oriented_nms.nms,
                            [single_batch_boxes, single_batch_scores, 
                            tf.constant(self._nms_iou_thresh), tf.constant(self._nms_size)],
                            tf.bool)
                        return single_batch_nms_mask

                    nms_masks = tf.map_fn(
                        single_batch_oriented_nms_fn,
                        elems=[bev_proposal_boxes, foreground_scores],
                        dtype=tf.bool)
                    nms_masks = tf.reshape(nms_masks, [self._batch_size, -1])
                
                    top_proposals = tf.reshape(
                                        tf.boolean_mask(proposals, nms_masks),
                                        [self._batch_size, -1, 7])
                    top_objectness_scores = tf.reshape(
                                            tf.boolean_mask(foreground_scores, nms_masks),
                                            [self._batch_size, -1])
                else:
                    def single_batch_nms_fn(args):
                        (single_batch_boxes_tf_order, single_batch_scores, single_batch_proposals) = args
                        
                        single_batch_nms_indices = tf.image.non_max_suppression(
                            single_batch_boxes_tf_order,
                            single_batch_scores,
                            max_output_size=self._nms_size,
                            iou_threshold=self._nms_iou_thresh)
                        
                        single_batch_top_proposals = tf.gather(single_batch_proposals,
                                                               single_batch_nms_indices)
                        single_batch_top_scores = tf.gather(single_batch_scores,
                                                            single_batch_nms_indices)
                        return [single_batch_top_proposals, 
                                single_batch_top_scores,
                                single_batch_nms_indices]

                    (top_proposals, top_objectness_scores, nms_indices) = tf.map_fn(
                        single_batch_nms_fn,
                        elems=[bev_proposal_boxes_tf_order, foreground_scores, proposals],
                        dtype=[tf.float32, tf.float32, tf.int32])
                
                tf.summary.histogram('top_objectness_scores', top_objectness_scores)
        
        top_anchors = tf.boolean_mask(top_anchors, non_empty_box)
        avod_projection_in = tf.boolean_mask(avod_projection_in, non_empty_box)
        ######################################################
        # Subsample mini_batch for the loss function
        ######################################################
        with tf.variable_scope('bev'):
            # Project top anchors into bev and image spaces
            bev_proposal_boxes, bev_proposal_boxes_norm = \
                anchor_projector.project_to_bev(
                    avod_projection_in,
                    self.dataset.kitti_utils.bev_extents)

            # Reorder projected boxes into [y1, x1, y2, x2]
            bev_proposal_boxes_tf_order = \
                anchor_projector.reorder_projected_boxes(
                    bev_proposal_boxes)
            bev_proposal_boxes_norm_tf_order = \
                anchor_projector.reorder_projected_boxes(
                    bev_proposal_boxes_norm)
        '''
        with tf.variable_scope('img'):
            image_shape = tf.cast(tf.shape(
                rpn_model.placeholders[RpnModel.PL_IMG_INPUT])[0:2],
                tf.float32)
            img_proposal_boxes, img_proposal_boxes_norm = \
                anchor_projector.tf_project_to_image_space(
                    avod_projection_in,
                    rpn_model.placeholders[RpnModel.PL_CALIB_P2],
                    image_shape)
            # Only reorder the normalized img
            img_proposal_boxes_norm_tf_order = \
                anchor_projector.reorder_projected_boxes(
                    img_proposal_boxes_norm)
        '''
        # Get the ground truth tensors
        anchors_gt = rpn_model.placeholders[RpnModel.PL_LABEL_ANCHORS]
        if self._box_rep in ['box_3d', 'box_4ca']:
            boxes_3d_gt = rpn_model.placeholders[RpnModel.PL_LABEL_BOXES_3D]
            orientations_gt = boxes_3d_gt[:, 6]
        elif self._box_rep in ['box_8c', 'box_8co', 'box_4c']:
            boxes_3d_gt = rpn_model.placeholders[RpnModel.PL_LABEL_BOXES_3D]
        else:
            raise NotImplementedError('Ground truth tensors not implemented')

        # Project anchor_gts to 2D bev
        with tf.variable_scope('avod_gt_projection'):
            bev_anchor_boxes_gt, _ = anchor_projector.project_to_bev(
                anchors_gt, self.dataset.kitti_utils.bev_extents)

            bev_anchor_boxes_gt_tf_order = \
                anchor_projector.reorder_projected_boxes(bev_anchor_boxes_gt)

        with tf.variable_scope('avod_box_list'):
            # Convert to box_list format
            anchor_box_list_gt = box_list.BoxList(bev_anchor_boxes_gt_tf_order)
            anchor_box_list = box_list.BoxList(bev_proposal_boxes_tf_order)

        mb_mask, mb_class_label_indices, mb_gt_indices = \
            self.sample_mini_batch(
                anchor_box_list_gt=anchor_box_list_gt,
                anchor_box_list=anchor_box_list,
                class_labels=class_labels)

        # Create classification one_hot vector
        with tf.variable_scope('avod_one_hot_classes'):
            mb_classification_gt = tf.one_hot(
                mb_class_label_indices,
                depth=self._num_final_classes,
                on_value=1.0 - self._config.label_smoothing_epsilon,
                off_value=(self._config.label_smoothing_epsilon /
                           self.dataset.num_classes))

        # TODO: Don't create a mini batch in test mode
        # Mask predictions
        with tf.variable_scope('avod_apply_mb_mask'):
            # Classification
            mb_classifications_logits = tf.boolean_mask(
                all_cls_logits, mb_mask)
            mb_classifications_softmax = tf.boolean_mask(
                all_cls_softmax, mb_mask)

            # Offsets
            mb_offsets = tf.boolean_mask(all_offsets, mb_mask)

            # Angle Vectors
            if all_angle_vectors is not None:
                mb_angle_vectors = tf.boolean_mask(all_angle_vectors, mb_mask)
            else:
                mb_angle_vectors = None

        # Encode anchor offsets
        with tf.variable_scope('avod_encode_mb_anchors'):
            mb_anchors = tf.boolean_mask(top_anchors, mb_mask)

            if self._box_rep == 'box_3d':
                # Gather corresponding ground truth anchors for each mb sample
                mb_anchors_gt = tf.gather(anchors_gt, mb_gt_indices)
                mb_offsets_gt = anchor_encoder.tf_anchor_to_offset(
                    mb_anchors, mb_anchors_gt)

                # Gather corresponding ground truth orientation for each
                # mb sample
                mb_orientations_gt = tf.gather(orientations_gt,
                                               mb_gt_indices)
            elif self._box_rep in ['box_8c', 'box_8co']:

                # Get boxes_3d ground truth mini-batch and convert to box_8c
                mb_boxes_3d_gt = tf.gather(boxes_3d_gt, mb_gt_indices)
                if self._box_rep == 'box_8c':
                    mb_boxes_8c_gt = \
                        box_8c_encoder.tf_box_3d_to_box_8c(mb_boxes_3d_gt)
                elif self._box_rep == 'box_8co':
                    mb_boxes_8c_gt = \
                        box_8c_encoder.tf_box_3d_to_box_8co(mb_boxes_3d_gt)

                # Convert proposals: anchors -> box_3d -> box8c
                proposal_boxes_3d = \
                    box_3d_encoder.anchors_to_box_3d(top_anchors, fix_lw=True)
                proposal_boxes_8c = \
                    box_8c_encoder.tf_box_3d_to_box_8c(proposal_boxes_3d)

                # Get mini batch offsets
                mb_boxes_8c = tf.boolean_mask(proposal_boxes_8c, mb_mask)
                mb_offsets_gt = box_8c_encoder.tf_box_8c_to_offsets(
                    mb_boxes_8c, mb_boxes_8c_gt)

                # Flatten the offsets to a (N x 24) vector
                mb_offsets_gt = tf.reshape(mb_offsets_gt, [-1, 24])

            elif self._box_rep in ['box_4c', 'box_4ca']:

                # Get ground plane for box_4c conversion
                ground_plane = self._rpn_model.placeholders[
                    self._rpn_model.PL_GROUND_PLANE]

                # Convert gt boxes_3d -> box_4c
                mb_boxes_3d_gt = tf.gather(boxes_3d_gt, mb_gt_indices)
                mb_boxes_4c_gt = box_4c_encoder.tf_box_3d_to_box_4c(
                    mb_boxes_3d_gt, ground_plane)

                # Convert proposals: anchors -> box_3d -> box_4c
                proposal_boxes_3d = \
                    box_3d_encoder.anchors_to_box_3d(top_anchors, fix_lw=True)
                proposal_boxes_4c = \
                    box_4c_encoder.tf_box_3d_to_box_4c(proposal_boxes_3d,
                                                       ground_plane)

                # Get mini batch
                mb_boxes_4c = tf.boolean_mask(proposal_boxes_4c, mb_mask)
                mb_offsets_gt = box_4c_encoder.tf_box_4c_to_offsets(
                    mb_boxes_4c, mb_boxes_4c_gt)

                if self._box_rep == 'box_4ca':
                    # Gather corresponding ground truth orientation for each
                    # mb sample
                    mb_orientations_gt = tf.gather(orientations_gt,
                                                   mb_gt_indices)

            else:
                raise NotImplementedError(
                    'Anchor encoding not implemented for', self._box_rep)

        ######################################################
        # ROI summary images
        ######################################################
        ''' 
        avod_mini_batch_size = \
            self.dataset.kitti_utils.mini_batch_utils.avod_mini_batch_size
        with tf.variable_scope('bev_avod_rois'):
            mb_bev_anchors_norm = tf.boolean_mask(
                bev_proposal_boxes_norm_tf_order, mb_mask)
            mb_bev_box_indices = tf.zeros_like(mb_gt_indices, dtype=tf.int32)

            # Show the ROIs of the BEV input density map
            # for the mini batch anchors
            bev_input_rois = tf.image.crop_and_resize(
                self._rpn_model._bev_preprocessed,
                mb_bev_anchors_norm,
                mb_bev_box_indices,
                (32, 32))

            bev_input_roi_summary_images = tf.split(
                bev_input_rois, self._bev_depth, axis=3)
            tf.summary.image('bev_avod_rois',
                             bev_input_roi_summary_images[-1],
                             max_outputs=avod_mini_batch_size)

        with tf.variable_scope('img_avod_rois'):
            # ROIs on image input
            mb_img_anchors_norm = tf.boolean_mask(
                img_proposal_boxes_norm_tf_order, mb_mask)
            mb_img_box_indices = tf.zeros_like(mb_gt_indices, dtype=tf.int32)

            # Do test ROI pooling on mini batch
            img_input_rois = tf.image.crop_and_resize(
                self._rpn_model._img_preprocessed,
                mb_img_anchors_norm,
                mb_img_box_indices,
                (32, 32))

            tf.summary.image('img_avod_rois',
                             img_input_rois,
                             max_outputs=avod_mini_batch_size)
        '''
        ######################################################
        # Final Predictions
        ######################################################
        # Get orientations from angle vectors
        if all_angle_vectors is not None:
            with tf.variable_scope('avod_orientation'):
                all_orientations = \
                    orientation_encoder.tf_angle_vector_to_orientation(
                        all_angle_vectors)

        # Apply offsets to regress proposals
        with tf.variable_scope('avod_regression'):
            if self._box_rep == 'box_3d':
                prediction_anchors = \
                    anchor_encoder.offset_to_anchor(top_anchors,
                                                    all_offsets)

            elif self._box_rep in ['box_8c', 'box_8co']:
                # Reshape the 24-dim regressed offsets to (N x 3 x 8)
                reshaped_offsets = tf.reshape(all_offsets,
                                              [-1, 3, 8])
                # Given the offsets, get the boxes_8c
                prediction_boxes_8c = \
                    box_8c_encoder.tf_offsets_to_box_8c(proposal_boxes_8c,
                                                        reshaped_offsets)
                # Convert corners back to box3D
                prediction_boxes_3d = \
                    box_8c_encoder.box_8c_to_box_3d(prediction_boxes_8c)

                # Convert the box_3d to anchor format for nms
                prediction_anchors = \
                    box_3d_encoder.tf_box_3d_to_anchor(prediction_boxes_3d)

            elif self._box_rep in ['box_4c', 'box_4ca']:
                # Convert predictions box_4c -> box_3d
                prediction_boxes_4c = \
                    box_4c_encoder.tf_offsets_to_box_4c(proposal_boxes_4c,
                                                        all_offsets)

                prediction_boxes_3d = \
                    box_4c_encoder.tf_box_4c_to_box_3d(prediction_boxes_4c,
                                                       ground_plane)

                # Convert to anchor format for nms
                prediction_anchors = \
                    box_3d_encoder.tf_box_3d_to_anchor(prediction_boxes_3d)

            else:
                raise NotImplementedError('Regression not implemented for',
                                          self._box_rep)

        # Apply Non-oriented NMS in BEV
        with tf.variable_scope('avod_nms'):
            bev_extents = self.dataset.kitti_utils.bev_extents

            with tf.variable_scope('bev_projection'):
                # Project predictions into BEV
                avod_bev_boxes, _ = anchor_projector.project_to_bev(
                    prediction_anchors, bev_extents)
                avod_bev_boxes_tf_order = \
                    anchor_projector.reorder_projected_boxes(
                        avod_bev_boxes)

            # Get top score from second column onward
            all_top_scores = tf.reduce_max(all_cls_logits[:, 1:], axis=1)

            # Apply NMS in BEV
            nms_indices = tf.image.non_max_suppression(
                avod_bev_boxes_tf_order,
                all_top_scores,
                max_output_size=self._nms_size,
                iou_threshold=self._nms_iou_threshold)

            # Gather predictions from NMS indices
            top_classification_logits = tf.gather(all_cls_logits,
                                                  nms_indices)
            top_classification_softmax = tf.gather(all_cls_softmax,
                                                   nms_indices)
            tf.summary.histogram('top_classification_scores', top_classification_softmax[:, 1])
            top_prediction_anchors = tf.gather(prediction_anchors,
                                               nms_indices)

            if self._box_rep == 'box_3d':
                top_orientations = tf.gather(
                    all_orientations, nms_indices)

            elif self._box_rep in ['box_8c', 'box_8co']:
                top_prediction_boxes_3d = tf.gather(
                    prediction_boxes_3d, nms_indices)
                top_prediction_boxes_8c = tf.gather(
                    prediction_boxes_8c, nms_indices)

            elif self._box_rep == 'box_4c':
                top_prediction_boxes_3d = tf.gather(
                    prediction_boxes_3d, nms_indices)
                top_prediction_boxes_4c = tf.gather(
                    prediction_boxes_4c, nms_indices)

            elif self._box_rep == 'box_4ca':
                top_prediction_boxes_3d = tf.gather(
                    prediction_boxes_3d, nms_indices)
                top_prediction_boxes_4c = tf.gather(
                    prediction_boxes_4c, nms_indices)
                top_orientations = tf.gather(
                    all_orientations, nms_indices)

            else:
                raise NotImplementedError('NMS gather not implemented for',
                                          self._box_rep)

        if self._train_val_test in ['train', 'val']:
            # Additional entries are added to the shared prediction_dict
            # Mini batch predictions
            prediction_dict[self.PRED_MB_CLASSIFICATION_LOGITS] = \
                mb_classifications_logits
            prediction_dict[self.PRED_MB_CLASSIFICATION_SOFTMAX] = \
                mb_classifications_softmax
            prediction_dict[self.PRED_MB_OFFSETS] = mb_offsets

            # Mini batch ground truth
            prediction_dict[self.PRED_MB_CLASSIFICATIONS_GT] = \
                mb_classification_gt
            prediction_dict[self.PRED_MB_OFFSETS_GT] = mb_offsets_gt

            # Top NMS predictions
            prediction_dict[self.PRED_TOP_CLASSIFICATION_LOGITS] = \
                top_classification_logits
            prediction_dict[self.PRED_TOP_CLASSIFICATION_SOFTMAX] = \
                top_classification_softmax

            prediction_dict[self.PRED_TOP_PREDICTION_ANCHORS] = \
                top_prediction_anchors

            # Mini batch predictions (for debugging)
            prediction_dict[self.PRED_MB_MASK] = mb_mask
            # prediction_dict[self.PRED_MB_POS_MASK] = mb_pos_mask
            prediction_dict[self.PRED_MB_CLASS_INDICES_GT] = \
                mb_class_label_indices

            # All predictions (for debugging)
            prediction_dict[self.PRED_ALL_CLASSIFICATIONS] = \
                all_cls_logits
            prediction_dict[self.PRED_ALL_OFFSETS] = all_offsets

            # Path drop masks (for debugging)
            # prediction_dict['bev_mask'] = bev_mask
            # prediction_dict['img_mask'] = img_mask

        else:
            # self._train_val_test == 'test'
            prediction_dict[self.PRED_TOP_CLASSIFICATION_SOFTMAX] = \
                top_classification_softmax
            prediction_dict[self.PRED_TOP_PREDICTION_ANCHORS] = \
                top_prediction_anchors

        if self._box_rep == 'box_3d':
            prediction_dict[self.PRED_MB_ANCHORS_GT] = mb_anchors_gt
            prediction_dict[self.PRED_MB_ORIENTATIONS_GT] = mb_orientations_gt
            prediction_dict[self.PRED_MB_ANGLE_VECTORS] = mb_angle_vectors

            prediction_dict[self.PRED_TOP_ORIENTATIONS] = top_orientations

            # For debugging
            prediction_dict[self.PRED_ALL_ANGLE_VECTORS] = all_angle_vectors

        elif self._box_rep in ['box_8c', 'box_8co']:
            prediction_dict[self.PRED_TOP_PREDICTION_BOXES_3D] = \
                top_prediction_boxes_3d

            # Store the corners before converting for visualization purposes
            prediction_dict[self.PRED_TOP_BOXES_8C] = top_prediction_boxes_8c

        elif self._box_rep == 'box_4c':
            prediction_dict[self.PRED_TOP_PREDICTION_BOXES_3D] = \
                top_prediction_boxes_3d
            prediction_dict[self.PRED_TOP_BOXES_4C] = top_prediction_boxes_4c

        elif self._box_rep == 'box_4ca':
            if self._train_val_test in ['train', 'val']:
                prediction_dict[self.PRED_MB_ORIENTATIONS_GT] = \
                    mb_orientations_gt
                prediction_dict[self.PRED_MB_ANGLE_VECTORS] = mb_angle_vectors

            prediction_dict[self.PRED_TOP_PREDICTION_BOXES_3D] = \
                top_prediction_boxes_3d
            prediction_dict[self.PRED_TOP_BOXES_4C] = top_prediction_boxes_4c
            prediction_dict[self.PRED_TOP_ORIENTATIONS] = top_orientations

        else:
            raise NotImplementedError('Prediction dict not implemented for',
                                      self._box_rep)

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
    
    def sample_mini_batch(self, anchor_box_list_gt, anchor_box_list,
                          class_labels):

        with tf.variable_scope('avod_create_mb_mask'):
            # Get IoU for every anchor
            all_ious = box_list_ops.iou(anchor_box_list_gt, anchor_box_list)
            max_ious = tf.reduce_max(all_ious, axis=0)
            max_iou_indices = tf.argmax(all_ious, axis=0)

            # Sample a pos/neg mini-batch from anchors with highest IoU match
            mini_batch_utils = self.dataset.kitti_utils.mini_batch_utils
            mb_mask, mb_pos_mask = mini_batch_utils.sample_avod_mini_batch(
                max_ious)
            mb_class_label_indices = mini_batch_utils.mask_class_label_indices(
                mb_pos_mask, mb_mask, max_iou_indices, class_labels)

            mb_gt_indices = tf.boolean_mask(max_iou_indices, mb_mask)

        return mb_mask, mb_class_label_indices, mb_gt_indices

    def create_feed_dict(self, batch_size=1):
        feed_dict = self._rpn_model.create_feed_dict(batch_size)
        self.sample_info = self._rpn_model.sample_info
        if self.model_config.alternating_training_step == 2:
            self._placeholder_inputs[self.PL_PROPOSALS] = self.dataset.load_proposals(self.sample_info['sample_name'])
        
            for key, value in self.placeholders.items():
                feed_dict[value] = self._placeholder_inputs[key]
        return feed_dict

    def loss(self, prediction_dict):
        # Note: The loss should be using mini-batch values only
        if self.model_config.alternating_training_step == 2:
            loss_dict = {}
            rpn_loss = tf.constant(0.0)
        else:
            loss_dict, rpn_loss = self._rpn_model.loss(prediction_dict)
        
        losses_output = avod_loss_builder.build(self, prediction_dict)

        classification_loss = \
            losses_output[avod_loss_builder.KEY_CLASSIFICATION_LOSS]

        final_reg_loss = losses_output[avod_loss_builder.KEY_REGRESSION_LOSS]

        avod_loss = losses_output[avod_loss_builder.KEY_AVOD_LOSS]

        offset_loss_norm = \
            losses_output[avod_loss_builder.KEY_OFFSET_LOSS_NORM]

        loss_dict.update({self.LOSS_FINAL_CLASSIFICATION: classification_loss})
        loss_dict.update({self.LOSS_FINAL_REGRESSION: final_reg_loss})

        # Add localization and orientation losses to loss dict for plotting
        loss_dict.update({self.LOSS_FINAL_LOCALIZATION: offset_loss_norm})

        ang_loss_norm = losses_output.get(
            avod_loss_builder.KEY_ANG_LOSS_NORM)
        if ang_loss_norm is not None:
            loss_dict.update({self.LOSS_FINAL_ORIENTATION: ang_loss_norm})

        with tf.variable_scope('total_loss'):
            total_loss = rpn_loss + avod_loss

        return loss_dict, total_loss
