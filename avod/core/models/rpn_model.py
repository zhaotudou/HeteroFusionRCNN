import numpy as np
import random
import tensorflow as tf
from tensorflow.contrib import slim

from avod.builders import feature_extractor_builder
from avod.core import anchor_encoder
from avod.core import anchor_filter
from avod.core import anchor_projector
from avod.core import box_3d_encoder
from avod.core import constants
from avod.core import losses
from avod.core import model
from avod.core import pointfly as pf
from avod.core import summary_utils
from avod.core.anchor_generators import grid_anchor_3d_generator
from avod.datasets.kitti import kitti_aug


class RpnModel(model.DetectionModel):
    ##############################
    # Keys for Placeholders
    ##############################
    PL_PC_INPUTS = 'pc_inputs_pl'
    #PL_IMG_INPUT = 'img_input_pl'
    #PL_ANCHORS = 'anchors_pl'

    '''
    PL_BEV_ANCHORS = 'bev_anchors_pl'
    PL_BEV_ANCHORS_NORM = 'bev_anchors_norm_pl'
    PL_IMG_ANCHORS = 'img_anchors_pl'
    PL_IMG_ANCHORS_NORM = 'img_anchors_norm_pl'
    '''
    PL_LABEL_SEGS = 'label_segs_pl'
    PL_LABEL_ANCHORS = 'label_anchors_pl'
    PL_LABEL_BOXES_3D = 'label_boxes_3d_pl'
    PL_LABEL_CLASSES = 'label_classes_pl'

    # Sample info, including keys for projection to image space
    # (e.g. camera matrix, image index, etc.)
    '''
    PL_CALIB_P2 = 'frame_calib_p2'
    PL_IMG_IDX = 'current_img_idx'
    '''

    ##############################
    # Keys for Predictions
    ##############################
    PRED_SEG_SOFTMAX = 'rpn_seg_softmax'
    PRED_SEG_GT = 'rpn_seg_gt'
    
    PRED_MB_OBJECTNESS_GT = 'rpn_mb_objectness_gt'
    PRED_MB_OFFSETS_GT = 'rpn_mb_offsets_gt'

    PRED_MB_MASK = 'rpn_mb_mask'
    PRED_MB_OBJECTNESS = 'rpn_mb_objectness'
    PRED_MB_OFFSETS = 'rpn_mb_offsets'

    PRED_TOP_INDICES = 'rpn_top_indices'
    PRED_TOP_ANCHORS = 'rpn_top_anchors'
    PRED_TOP_OBJECTNESS_SOFTMAX = 'rpn_top_objectness_softmax'

    ##############################
    # Keys for Loss
    ##############################
    LOSS_RPN_SEGMENTATION = 'rpn_segmentation_loss'
    LOSS_RPN_OBJECTNESS = 'rpn_objectness_loss'
    LOSS_RPN_REGRESSION = 'rpn_regression_loss'

    def __init__(self, model_config, train_val_test, dataset):
        """
        Args:
            model_config: configuration for the model
            train_val_test: "train", "val", or "test"
            dataset: the dataset that will provide samples and ground truth
        """

        # Sets model configs (_config)
        super(RpnModel, self).__init__(model_config)

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

        #self._img_pixel_size = np.asarray([input_config.img_dims_h,
        #                                   input_config.img_dims_w])
        #self._img_depth = input_config.img_depth

        # Rpn config
        rpn_config = self._config.rpn_config
        self._proposal_roi_crop_size = rpn_config.rpn_proposal_roi_crop_size
        self._fusion_method = rpn_config.rpn_fusion_method

        if self._train_val_test in ["train", "val"]:
            self._nms_size = rpn_config.rpn_train_nms_size
        else:
            self._nms_size = rpn_config.rpn_test_nms_size

        self._nms_iou_thresh = rpn_config.rpn_nms_iou_thresh

        # Feature Extractor Nets
        self._pc_feature_extractor = \
            feature_extractor_builder.get_extractor(
                self._config.layers_config.pc_feature_extractor)
        '''
        self._img_feature_extractor = \
            feature_extractor_builder.get_extractor(
                self._config.layers_config.img_feature_extractor)
        '''
        # Network input placeholders
        self.placeholders = dict()

        # Inputs to network placeholders
        self._placeholder_inputs = dict()

        # Information about the current sample
        self.samples_info = []

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
        
        #self._path_drop_probabilities = self._config.path_drop_probabilities
        self._train_on_all_samples = self._config.train_on_all_samples
        self._eval_all_samples = self._config.eval_all_samples
        # Overwrite the dataset's variable with the config
        self.dataset.train_on_all_samples = self._train_on_all_samples
        self.dataset.eval_all_samples = self._eval_all_samples
        '''
        if self._train_val_test in ["val", "test"]:
            # Disable path-drop, this should already be disabled inside the
            # evaluator, but just in case.
            self._path_drop_probabilities[0] = 1.0
            self._path_drop_probabilities[1] = 1.0
        '''

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
                                        tf.float32, (None, None, self._pc_data_dim),
                                        self.PL_PC_INPUTS)

            self._pc_pts_preprocessed, self._pc_fts_preprocessed = \
                self._pc_feature_extractor.preprocess_input(
                    pc_input_placeholder, 
                    self._config.input_config,
                    self._is_training)
        
        with tf.variable_scope('pl_labels'):
            self._add_placeholder(tf.int32, [None, None], 
                                  self.PL_LABEL_SEGS)
            #self._add_placeholder(tf.float32, [None, 6],
            #                      self.PL_LABEL_ANCHORS)
            #self._add_placeholder(tf.float32, [None, 7],
            #                      self.PL_LABEL_BOXES_3D)
            #self._add_placeholder(tf.float32, [None],
            #                      self.PL_LABEL_CLASSES)

    def _set_up_feature_extractors(self):
        """Sets up feature extractors and stores feature maps and
        bottlenecks as member variables.
        """

        #self.bev_feature_maps, self.bev_end_points = \
        self._pc_pts, self._pc_fts = \
            self._pc_feature_extractor.build(
                self._pc_pts_preprocessed,
                self._pc_fts_preprocessed,
                self._is_training)
        tf.summary.histogram('pc_fts', self._pc_fts)

    def build(self):

        # Setup input placeholders
        self._set_up_input_pls()

        # Setup feature extractors
        self._set_up_feature_extractors()

        seg_logits = pf.dense(self._pc_fts, self.num_classes + 1, 'seg_logits', 
                              self._is_training, with_bn=False, activation=None)
        seg_softmax = tf.nn.softmax(seg_logits)
        '''
        with tf.variable_scope('anchor_predictor', 'ap', [rpn_fusion_out]):
            tensor_in = rpn_fusion_out
            tensor_in = tf.expand_dims(tensor_in, 2)

            # Parse rpn layers config
            layers_config = self._config.layers_config.rpn_config
            l2_weight_decay = layers_config.l2_weight_decay

            if l2_weight_decay > 0:
                weights_regularizer = slim.l2_regularizer(l2_weight_decay)
            else:
                weights_regularizer = None

            with slim.arg_scope([slim.conv2d],
                                weights_regularizer=weights_regularizer):
                # Use conv2d instead of fully_connected layers.
                cls_fc6 = slim.conv2d(tensor_in,
                                      layers_config.cls_fc6,
                                      [self._proposal_roi_crop_size, 1],
                                      padding='VALID',
                                      scope='cls_fc6')

                cls_fc6_drop = slim.dropout(cls_fc6,
                                            layers_config.keep_prob,
                                            is_training=self._is_training,
                                            scope='cls_fc6_drop')

                cls_fc7 = slim.conv2d(cls_fc6_drop,
                                      layers_config.cls_fc7,
                                      [1, 1],
                                      scope='cls_fc7')

                cls_fc7_drop = slim.dropout(cls_fc7,
                                            layers_config.keep_prob,
                                            is_training=self._is_training,
                                            scope='cls_fc7_drop')

                cls_fc8 = slim.conv2d(cls_fc7_drop,
                                      2,
                                      [1, 1],
                                      activation_fn=None,
                                      scope='cls_fc8')

                objectness = tf.squeeze(
                    cls_fc8, [1, 2],
                    name='cls_fc8/squeezed')

                # Use conv2d instead of fully_connected layers.
                reg_fc6 = slim.conv2d(tensor_in,
                                      layers_config.reg_fc6,
                                      [self._proposal_roi_crop_size, 1],
                                      padding='VALID',
                                      scope='reg_fc6')

                reg_fc6_drop = slim.dropout(reg_fc6,
                                            layers_config.keep_prob,
                                            is_training=self._is_training,
                                            scope='reg_fc6_drop')

                reg_fc7 = slim.conv2d(reg_fc6_drop,
                                      layers_config.reg_fc7,
                                      [1, 1],
                                      scope='reg_fc7')

                reg_fc7_drop = slim.dropout(reg_fc7,
                                            layers_config.keep_prob,
                                            is_training=self._is_training,
                                            scope='reg_fc7_drop')

                reg_fc8 = slim.conv2d(reg_fc7_drop,
                                      6,
                                      [1, 1],
                                      activation_fn=None,
                                      scope='reg_fc8')

                offsets = tf.squeeze(
                    reg_fc8, [1, 2],
                    name='reg_fc8/squeezed')

        with tf.variable_scope('histograms_rpn'):
            with tf.variable_scope('anchor_predictor'):
                fc_layers = [cls_fc6, cls_fc7, cls_fc8, objectness,
                             reg_fc6, reg_fc7, reg_fc8, offsets]
                for fc_layer in fc_layers:
                    # fix the name to avoid tf warnings
                    tf.summary.histogram(fc_layer.name.replace(':', '_'),
                                         fc_layer)

        # Return the proposals
        with tf.variable_scope('proposals'):
            anchors = self.placeholders[self.PL_ANCHORS]
            non_empty_anchors = tf.boolean_mask(anchors, non_empty_box)

            # Decode anchor regression offsets
            with tf.variable_scope('decoding'):
                regressed_anchors = anchor_encoder.offset_to_anchor(
                        non_empty_anchors, offsets)
            
            with tf.variable_scope('bev_projection'):
                _, bev_proposal_boxes_norm = anchor_projector.project_to_bev(
                    regressed_anchors, self._bev_extents)
            
            with tf.variable_scope('softmax'):
                objectness_softmax = tf.nn.softmax(objectness)

            with tf.variable_scope('nms'):
                objectness_scores = objectness_softmax[:, 1]

                # Do NMS on regressed anchors
                top_indices = tf.image.non_max_suppression(
                    bev_proposal_boxes_norm, objectness_scores,
                    max_output_size=self._nms_size,
                    iou_threshold=self._nms_iou_thresh)

                top_anchors = tf.gather(regressed_anchors, top_indices)
                top_objectness_softmax = tf.gather(objectness_scores,
                                                   top_indices)
                tf.summary.histogram('top_objectness_scores', top_objectness_softmax)
                # top_offsets = tf.gather(offsets, top_indices)
                # top_objectness = tf.gather(objectness, top_indices)

        # Get mini batch
        all_ious_gt = self.placeholders[self.PL_ANCHOR_IOUS]
        all_offsets_gt = self.placeholders[self.PL_ANCHOR_OFFSETS]
        all_classes_gt = self.placeholders[self.PL_ANCHOR_CLASSES]

        all_ious_gt = tf.boolean_mask(all_ious_gt, non_empty_box)
        all_offsets_gt = tf.boolean_mask(all_offsets_gt, non_empty_box)
        all_classes_gt = tf.boolean_mask(all_classes_gt, non_empty_box)

        with tf.variable_scope('mini_batch'):
            mini_batch_utils = self.dataset.kitti_utils.mini_batch_utils
            mini_batch_mask, _ = \
                mini_batch_utils.sample_rpn_mini_batch(all_ious_gt)
        ''' 
        # Ground Truth Tensors
        with tf.variable_scope('one_hot_classes'):

            label_segs = self.placeholders[self.PL_LABEL_SEGS]
            segs_gt_one_hot = tf.one_hot(
                label_segs, depth=self.num_classes + 1,
                on_value=1.0,
                off_value=0.0)
            
            # Anchor classification ground truth
            # Object / Not Object
            '''
            min_pos_iou = \
                self.dataset.kitti_utils.mini_batch_utils.rpn_pos_iou_range[0]

            objectness_classes_gt = tf.cast(
                tf.greater_equal(all_ious_gt, min_pos_iou),
                dtype=tf.int32)
            objectness_gt = tf.one_hot(
                objectness_classes_gt, depth=2,
                on_value=1.0 - self._config.label_smoothing_epsilon,
                off_value=self._config.label_smoothing_epsilon)

        # Mask predictions for mini batch
        with tf.variable_scope('prediction_mini_batch'):
            objectness_masked = tf.boolean_mask(objectness, mini_batch_mask)
            offsets_masked = tf.boolean_mask(offsets, mini_batch_mask)

        with tf.variable_scope('ground_truth_mini_batch'):
            objectness_gt_masked = tf.boolean_mask(
                objectness_gt, mini_batch_mask)
            offsets_gt_masked = tf.boolean_mask(all_offsets_gt,
                                                mini_batch_mask)
        '''
        # Specify the tensors to evaluate
        predictions = dict()

        # Temporary predictions for debugging
        # predictions['anchor_ious'] = anchor_ious
        # predictions['anchor_offsets'] = all_offsets_gt

        if self._train_val_test in ['train', 'val']:
            predictions[self.PRED_SEG_SOFTMAX] = seg_softmax
            predictions[self.PRED_SEG_GT] = segs_gt_one_hot
            ''' 
            # Mini-batch masks
            predictions[self.PRED_MB_MASK] = mini_batch_mask
            # Mini-batch predictions
            predictions[self.PRED_MB_OBJECTNESS] = objectness_masked
            predictions[self.PRED_MB_OFFSETS] = offsets_masked

            # Mini batch ground truth
            predictions[self.PRED_MB_OFFSETS_GT] = offsets_gt_masked
            predictions[self.PRED_MB_OBJECTNESS_GT] = objectness_gt_masked

            # Proposals after nms
            predictions[self.PRED_TOP_INDICES] = top_indices
            predictions[self.PRED_TOP_ANCHORS] = top_anchors
            predictions[
                self.PRED_TOP_OBJECTNESS_SOFTMAX] = top_objectness_softmax
            
        else:
            # self._train_val_test == 'test'
            predictions[self.PRED_TOP_ANCHORS] = top_anchors
            predictions[
                self.PRED_TOP_OBJECTNESS_SOFTMAX] = top_objectness_softmax
        '''
        return predictions

    def create_feed_dict(self, batch_size=16, sample_index=None):
        """ Fills in the placeholders with the actual input values.
            Currently, only a batch size of 1 is supported

        Args:
            sample_index: optional, only used when train_val_test == 'test',
                a particular sample index in the dataset
                sample list to build the feed_dict for

        Returns:
            a feed_dict dictionary that can be used in a tensorflow session
        """

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
                samples = self.dataset.next_batch(batch_size=1, shuffle=False)
        
        if self._is_training:
            offset = int(random.gauss(0, self._pc_sample_pts * self._pc_sample_pts_variance))
            offset = max(offset, -self._pc_sample_pts * self._pc_sample_pts_clip)
            offset = min(offset, self._pc_sample_pts * self._pc_sample_pts_clip)
            pc_sample_pts = int(self._pc_sample_pts + offset)
        else:
            pc_sample_pts = self._pc_sample_pts
        
        pc_inputs = []
        label_segs = []
        self.samples_info.clear()
        for sample in samples:
            # Get ground truth data
            label_anchors = sample.get(constants.KEY_LABEL_ANCHORS)
            label_classes = sample.get(constants.KEY_LABEL_CLASSES)
            # We only need orientation from box_3d
            label_boxes_3d = sample.get(constants.KEY_LABEL_BOXES_3D)

            # Network input data
            # image_input = sample.get(constants.KEY_IMAGE_INPUT)
            # Image shape (h, w)
            # image_shape = [image_input.shape[0], image_input.shape[1]]

            # bev_input = sample.get(constants.KEY_PC_INPUT)
            pc_input = sample.get(constants.KEY_POINT_CLOUD)
            label_seg = sample.get(constants.KEY_LABEL_SEG)
            pool_size = pc_input.shape[0]
            if pool_size > pc_sample_pts:
                choices = np.random.choice(pool_size, pc_sample_pts, replace=False)
            else:
                choices = np.concatenate((np.random.choice(pool_size, pool_size, replace=False),
                                          np.random.choice(pool_size, pc_sample_pts - pool_size, replace=True)))
            pc_input = pc_input[choices]
            label_seg = label_seg[choices]
            
            pc_inputs.append(pc_input)
            label_segs.append(label_seg)
            
            # Temporary sample info for debugging
            sample_name = sample.get(constants.KEY_SAMPLE_NAME)
            self.samples_info.append(sample_name)

        # ground_plane = sample.get(constants.KEY_GROUND_PLANE)
        # stereo_calib_p2 = sample.get(constants.KEY_STEREO_CALIB_P2)

        # Fill the placeholders for anchor information
        # self._fill_anchor_pl_inputs(anchors_info=anchors_info,
        #                            ground_plane=ground_plane,
        #                            pc_input=pc_input,
        #                            image_shape=image_shape,
        #                            sample_name=sample_name,
        #                            sample_augs=sample_augs)

        # this is a list to match the explicit shape for the placeholder
        # self._placeholder_inputs[self.PL_IMG_IDX] = [int(sample_name)]

        # Fill in the rest
        # self._placeholder_inputs[self.PL_BEV_INPUT] = bev_input
        # self._placeholder_inputs[self.PL_IMG_INPUT] = image_input
        self._placeholder_inputs[self.PL_PC_INPUTS] = np.asarray(pc_inputs)
        self._placeholder_inputs[self.PL_LABEL_SEGS] = np.asarray(label_segs)
        
        #self._placeholder_inputs[self.PL_LABEL_ANCHORS] = label_anchors
        #self._placeholder_inputs[self.PL_LABEL_BOXES_3D] = label_boxes_3d
        #self._placeholder_inputs[self.PL_LABEL_CLASSES] = label_classes

        # Sample Info
        # img_idx is a list to match the placeholder shape
        # self._placeholder_inputs[self.PL_IMG_IDX] = [int(sample_name)]
        # self._placeholder_inputs[self.PL_CALIB_P2] = stereo_calib_p2
        # self._placeholder_inputs[self.PL_GROUND_PLANE] = ground_plane

        # Create a feed_dict and fill it with input values
        feed_dict = dict()
        for key, value in self.placeholders.items():
            feed_dict[value] = self._placeholder_inputs[key]

        return feed_dict

    def _fill_anchor_pl_inputs(self,
                               anchors_info,
                               ground_plane,
                               pc_input,
                               image_shape,
                               sample_name,
                               sample_augs):
        """
        Fills anchor placeholder inputs with corresponding data

        Args:
            anchors_info: anchor info from mini_batch_utils
            ground_plane: ground plane coefficients
            image_shape: image shape (h, w), used for projecting anchors
            sample_name: name of the sample, e.g. "000001"
            sample_augs: list of sample augmentations
        """

        # Lists for merging anchors info
        all_anchor_boxes_3d = []
        anchor_ious = []
        anchor_offsets = []
        anchor_classes = []

        # Create anchors for each class
        if len(self.dataset.classes) > 1:
            for class_idx in range(len(self.dataset.classes)):
                # Generate anchors for all classes
                grid_anchor_boxes_3d = self._anchor_generator.generate(
                    area_3d=self._area_extents,
                    anchor_3d_sizes=self._cluster_sizes[class_idx],
                    anchor_stride=self._anchor_strides[class_idx],
                    ground_plane=ground_plane)
                all_anchor_boxes_3d.append(grid_anchor_boxes_3d)
            all_anchor_boxes_3d = np.concatenate(all_anchor_boxes_3d)
        else:
            # Don't loop for a single class
            class_idx = 0
            grid_anchor_boxes_3d = self._anchor_generator.generate(
                area_3d=self._area_extents,
                anchor_3d_sizes=self._cluster_sizes[class_idx],
                anchor_stride=self._anchor_strides[class_idx],
                ground_plane=ground_plane)
            all_anchor_boxes_3d = grid_anchor_boxes_3d
        
        # Filter empty anchors
        # Skip if anchors_info is []
        sample_has_labels = True
        if self._train_val_test in ['train', 'val']:
            # Read in anchor info during training / validation
            if anchors_info:
                anchor_indices, anchor_ious, anchor_offsets, \
                    anchor_classes = anchors_info

                anchor_boxes_3d_to_use = all_anchor_boxes_3d[anchor_indices]
            else:
                train_cond = (self._train_val_test == "train" and
                              self._train_on_all_samples)
                eval_cond = (self._train_val_test == "val" and
                             self._eval_all_samples)
                if train_cond or eval_cond:
                    sample_has_labels = False
        else:
            sample_has_labels = False

        if not sample_has_labels:
            # During testing, or validation with no anchor info, manually
            # filter empty anchors
            # TODO: share voxel_grid_2d with BEV generation if possible
            voxel_grid_2d = \
                self.dataset.kitti_utils.create_sliced_voxel_grid_2d(
                    sample_name, pc_input)

            # Convert to anchors and filter
            anchors_to_use = box_3d_encoder.box_3d_to_anchor(
                all_anchor_boxes_3d)
            empty_filter = anchor_filter.get_empty_anchor_filter_2d(
                anchors_to_use, voxel_grid_2d, density_threshold=1)

            anchor_boxes_3d_to_use = all_anchor_boxes_3d[empty_filter]

        # Convert lists to ndarrays
        anchor_boxes_3d_to_use = np.asarray(anchor_boxes_3d_to_use)
        anchor_ious = np.asarray(anchor_ious)
        anchor_offsets = np.asarray(anchor_offsets)
        anchor_classes = np.asarray(anchor_classes)

        # Flip anchors and centroid x offsets for augmented samples
        if kitti_aug.AUG_FLIPPING in sample_augs:
            anchor_boxes_3d_to_use = kitti_aug.flip_boxes_3d(
                anchor_boxes_3d_to_use, flip_ry=False)
            if anchors_info:
                anchor_offsets[:, 0] = -anchor_offsets[:, 0]

        # Convert to anchors
        anchors_to_use = box_3d_encoder.box_3d_to_anchor(
            anchor_boxes_3d_to_use)
        num_anchors = len(anchors_to_use)
        '''
        # Project anchors into bev
        bev_anchors, bev_anchors_norm = anchor_projector.project_to_bev(
            anchors_to_use, self._bev_extents)

        # Project box_3d anchors into image space
        img_anchors, img_anchors_norm = \
            anchor_projector.project_to_image_space(
                anchors_to_use, stereo_calib_p2, image_shape)

        # Reorder into [y1, x1, y2, x2] for tf.crop_and_resize op
        self._bev_anchors_norm = bev_anchors_norm[:, [1, 0, 3, 2]]
        self._img_anchors_norm = img_anchors_norm[:, [1, 0, 3, 2]]
        '''
        # Fill in placeholder inputs
        self._placeholder_inputs[self.PL_ANCHORS] = anchors_to_use

        # If we are in train/validation mode, and the anchor infos
        # are not empty, store them. Checking for just anchor_ious
        # to be non-empty should be enough.
        if self._train_val_test in ['train', 'val'] and \
                len(anchor_ious) > 0:
            self._placeholder_inputs[self.PL_ANCHOR_IOUS] = anchor_ious
            self._placeholder_inputs[self.PL_ANCHOR_OFFSETS] = anchor_offsets
            self._placeholder_inputs[self.PL_ANCHOR_CLASSES] = anchor_classes

        # During test, or val when there is no anchor info
        elif self._train_val_test in ['test'] or \
                len(anchor_ious) == 0:
            # During testing, or validation with no gt, fill these in with 0s
            self._placeholder_inputs[self.PL_ANCHOR_IOUS] = \
                np.zeros(num_anchors)
            self._placeholder_inputs[self.PL_ANCHOR_OFFSETS] = \
                np.zeros([num_anchors, 6])
            self._placeholder_inputs[self.PL_ANCHOR_CLASSES] = \
                np.zeros(num_anchors)
        else:
            raise ValueError('Got run mode {}, and non-empty anchor info'.
                             format(self._train_val_test))
        '''
        self._placeholder_inputs[self.PL_BEV_ANCHORS] = bev_anchors
        self._placeholder_inputs[self.PL_BEV_ANCHORS_NORM] = \
            self._bev_anchors_norm
        self._placeholder_inputs[self.PL_IMG_ANCHORS] = img_anchors
        self._placeholder_inputs[self.PL_IMG_ANCHORS_NORM] = \
            self._img_anchors_norm
        '''

    def loss(self, prediction_dict):

        seg_softmax = prediction_dict[self.PRED_SEG_SOFTMAX]
        seg_gt = prediction_dict[self.PRED_SEG_GT]
        '''
        # these should include mini-batch values only
        objectness_gt = prediction_dict[self.PRED_MB_OBJECTNESS_GT]
        offsets_gt = prediction_dict[self.PRED_MB_OFFSETS_GT]

        # Predictions
        with tf.variable_scope('rpn_prediction_mini_batch'):
            objectness = prediction_dict[self.PRED_MB_OBJECTNESS]
            offsets = prediction_dict[self.PRED_MB_OFFSETS]
        '''
        with tf.variable_scope('rpn_losses'):
            with tf.variable_scope('segmentation'):
                seg_loss = losses.WeightedFocalLoss()
                seg_loss_weight = self._config.loss_config.seg_loss_weight
                segmentation_loss = seg_loss(
                            seg_softmax,
                            seg_gt,
                            weight=seg_loss_weight)
                tf.summary.scalar('segmentation', segmentation_loss)
            '''
            with tf.variable_scope('objectness'):
                cls_loss = losses.WeightedSoftmaxLoss()
                cls_loss_weight = self._config.loss_config.cls_loss_weight
                objectness_loss = cls_loss(objectness,
                                           objectness_gt,
                                           weight=cls_loss_weight)

                with tf.variable_scope('obj_norm'):
                    # normalize by the number of anchor mini-batches
                    objectness_loss = objectness_loss / tf.cast(
                        tf.shape(objectness_gt)[0], dtype=tf.float32)
                    tf.summary.scalar('objectness', objectness_loss)

            with tf.variable_scope('regression'):
                reg_loss = losses.WeightedSmoothL1Loss()
                reg_loss_weight = self._config.loss_config.reg_loss_weight
                anchorwise_localization_loss = reg_loss(offsets,
                                                        offsets_gt,
                                                        weight=reg_loss_weight)
                masked_localization_loss = \
                    anchorwise_localization_loss * objectness_gt[:, 1]
                localization_loss = tf.reduce_sum(masked_localization_loss)

                with tf.variable_scope('reg_norm'):
                    # normalize by the number of positive objects
                    num_positives = tf.reduce_sum(objectness_gt[:, 1])
                    # Assert the condition `num_positives > 0`
                    with tf.control_dependencies(
                            [tf.assert_positive(num_positives)]):
                        localization_loss = localization_loss / num_positives
                        tf.summary.scalar('regression', localization_loss)
            '''
            with tf.variable_scope('rpn_loss'):
                rpn_loss = segmentation_loss
                #rpn_loss = segmentation_loss + objectness_loss + localization_loss

        loss_dict = {
            self.LOSS_RPN_SEGMENTATION: segmentation_loss,
            #self.LOSS_RPN_OBJECTNESS: objectness_loss,
            #self.LOSS_RPN_REGRESSION: localization_loss,
        }

        return loss_dict, rpn_loss
