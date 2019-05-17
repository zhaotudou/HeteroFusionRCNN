import numpy as np

import tensorflow as tf

from avod.builders import feature_extractor_builder
from avod.builders import avod_fc_layers_builder
from avod.builders import avod_loss_builder
from avod.core import anchor_projector
from avod.core import box_3d_projector
from avod.core import anchor_encoder
from avod.core import box_3d_encoder
from avod.core import box_8c_encoder
from avod.core import box_4c_encoder
from avod.core import bin_based_box3d_encoder
from avod.core import pointfly as pf
from avod.core import compute_iou

from avod.core import box_util
from avod.core import constants

from avod.core import model
from avod.core import losses


class AvodModel(model.DetectionModel):
    ##############################
    # Keys for Placeholders
    ##############################
    PL_PROPOSALS = "proposals_pl"
    PL_PROPOSALS_IOU = "proposals_iou_pl"
    PL_PROPOSALS_GT = "proposals_gt_pl"

    PL_RPN_PTS = "rpn_pts_pl"
    PL_RPN_INTENSITY = "rpn_intensity_pl"
    PL_RPN_FG_MASK = "rpn_fg_mask_pl"
    PL_RPN_FTS = "rpn_fts_pl"
    ##############################
    # Keys for Predictions
    ##############################
    # Mini batch (mb) cls
    PRED_MB_CLASSIFICATION_LOGITS = "avod_mb_classification_logits"
    PRED_MB_CLASSIFICATIONS_GT = "avod_mb_classifications_gt"
    PRED_MB_CLASSIFICATION_MASK = "avod_mb_classification_mask"

    # Mini batch (mb) cls-reg
    PRED_MB_CLS = "avod_mb_cls"
    PRED_MB_REG = "avod_mb_reg"
    PRED_MB_CLS_GT = "avod_mb_cls_gt"
    PRED_MB_REG_GT = "avod_mb_reg_gt"
    PRED_MB_POS_REG_MASK = "avod_mb_pos_reg_mask"

    # predictions & BEV NMS
    PRED_BOXES = "avod_prediction_boxes"
    PRED_SOFTMAX = "avod_prediction_softmax"
    PRED_NON_EMPTY_BOX_MASK = "avod_prediction_non_empty_box_mask"
    PRED_NMS_INDICES = "avod_prediction_nms_indices"

    ##############################
    # Keys for Loss
    ##############################
    LOSS_FINAL_CLASSIFICATION = "avod_classification_loss"
    LOSS_FINAL_BIN_CLASSIFICATION = "avod_bin_classification_loss"
    LOSS_FINAL_REGRESSION = "avod_regression_loss"

    # (for debugging)
    LOSS_FINAL_ORIENTATION = "avod_orientation_loss"
    LOSS_FINAL_LOCALIZATION = "avod_localization_loss"

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

        self.dataset = dataset
        self._bev_extents = self.dataset.kitti_utils.bev_extents
        self._cluster_sizes, _ = self.dataset.get_cluster_info()

        # Dataset config
        self.num_classes = dataset.num_classes

        # Input config
        input_config = self._config.input_config
        self._pc_sample_pts = input_config.pc_sample_pts

        # self._img_pixel_size = np.asarray([input_config.img_dims_h,
        #                                   input_config.img_dims_w])
        # self._img_depth = [input_config.img_depth]

        # AVOD config
        avod_config = self._config.avod_config
        self._use_intensity_feature = avod_config.avod_use_intensity_feature
        self._proposal_roi_crop_size = avod_config.avod_proposal_roi_crop_size
        self._positive_selection = avod_config.avod_positive_selection
        self._nms_size = avod_config.avod_nms_size
        self._nms_iou_thresh = avod_config.avod_nms_iou_thresh
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
        self._pc_feature_extractor = feature_extractor_builder.get_extractor(
            self._config.layers_config.avod_config.pc_feature_extractor
        )
        if self._box_rep not in ["box_3d", "box_8c", "box_8co", "box_4c", "box_4ca"]:
            raise ValueError("Invalid box representation", self._box_rep)

        if train_val_test not in ["train", "val", "test"]:
            raise ValueError(
                "Invalid train_val_test value,"
                'should be one of ["train", "val", "test"]'
            )
        self._train_val_test = train_val_test
        self._is_training = self._train_val_test == "train"
        self.dataset.train_val_test = self._train_val_test

        # Network input placeholders
        self.placeholders = dict()

        # Inputs to network placeholders
        self._placeholder_inputs = dict()

        self._sample_names = []

    def _add_placeholder(self, dtype, shape, name):
        placeholder = tf.placeholder(dtype, shape, name)
        self.placeholders[name] = placeholder
        return placeholder

    def _set_up_input_pls(self):
        """Sets up input placeholders by adding them to self._placeholders.
        Keys are defined as self.PL_*.
        """
        with tf.variable_scope("pl_proposals"):
            self._add_placeholder(
                tf.float32, [self._batch_size, None, 7], self.PL_PROPOSALS
            )
            self._add_placeholder(
                tf.float32, [self._batch_size, None], self.PL_PROPOSALS_IOU
            )
            self._add_placeholder(
                tf.float32, [self._batch_size, None, 8], self.PL_PROPOSALS_GT
            )

        with tf.variable_scope("pl_rpn_feature"):
            self._add_placeholder(
                tf.float32, [self._batch_size, self._pc_sample_pts, 3], self.PL_RPN_PTS
            )
            self._add_placeholder(
                tf.float32,
                [self._batch_size, self._pc_sample_pts],
                self.PL_RPN_INTENSITY,
            )
            self._add_placeholder(
                tf.bool, [self._batch_size, self._pc_sample_pts], self.PL_RPN_FG_MASK
            )
            # TODO: rm channel size hard coding
            self._add_placeholder(
                tf.float32,
                [self._batch_size, self._pc_sample_pts, 256],
                self.PL_RPN_FTS,
            )

    @classmethod
    def _canonical_transform(cls, pts, boxes_3d):
        """
        Canonical Coordinate Transform
        Input:
            pts: (N,R,3) [x,y,z] float32
            boxes_3d:(N,7)  [cx,cy,cz,l,w,h,ry] float 32
        Output:
            pts_ct: (N,R,3) [x',y',z'] float32
        """
        pts_shift = pts - tf.expand_dims(boxes_3d[:, 0:3], 1)

        all_rys = boxes_3d[:, 6] * -1
        ry_sin = tf.sin(all_rys)
        ry_cos = tf.cos(all_rys)

        zeros = tf.zeros_like(all_rys, dtype=tf.float32)
        ones = tf.ones_like(all_rys, dtype=tf.float32)

        # Rotation matrix
        rot_mats = tf.stack(
            [
                tf.stack([ry_cos, zeros, ry_sin], axis=1),
                tf.stack([zeros, ones, zeros], axis=1),
                tf.stack([-ry_sin, zeros, ry_cos], axis=1),
            ],
            axis=2,
        )
        pts_rot = tf.matmul(rot_mats, pts_shift, transpose_a=True, transpose_b=True)
        return tf.matrix_transpose(pts_rot)

    def _gather_residuals(
        self, res_x_norms, res_z_norms, res_theta_norms, bin_x, bin_z, bin_theta
    ):

        """
        Input:
            res_x_norms: (N,K)
            bin_x:(N)
        return:
            res_x_norm: (N)
        """

        """
        #TF version: (if N is not None)
        ##########
        N = bin_x.shape[0].value
        Ns = tf.reshape(tf.range(N), [N,1])

        NK_x = tf.concat([Ns, tf.reshape(bin_x, [N,1])], axis=1) # (N,2)
        res_x_norm = tf.gather_nd(res_x_norms, NK_x) #(N)
        
        NK_z = tf.concat([Ns, tf.reshape(bin_z, [N,1])], axis=1) # (N,2)
        res_z_norm = tf.gather_nd(res_z_norms, NK_z) #(N)
        
        NK_theta = tf.concat([Ns, tf.reshape(bin_theta, [N,1])], axis=1) # (N,2)
        res_theta_norm = tf.gather_nd(res_theta_norms, NK_theta) #(N)
    
        """
        # NumPy version: if N is None, by using tf.py_func, N should be determined
        #############
        res_x_norm = np.take_along_axis(
            res_x_norms, np.expand_dims(bin_x, -1), axis=-1
        )  # (N,1)
        res_x_norm = np.squeeze(res_x_norm, -1)

        res_z_norm = np.take_along_axis(
            res_z_norms, np.expand_dims(bin_z, -1), axis=-1
        )  # (N,1)
        res_z_norm = np.squeeze(res_z_norm, -1)

        res_theta_norm = np.take_along_axis(
            res_theta_norms, np.expand_dims(bin_theta, -1), axis=-1
        )  # (N,1)
        res_theta_norm = np.squeeze(res_theta_norm, -1)

        return res_x_norm, res_z_norm, res_theta_norm

    def _gather_mean_sizes(self, cluster_sizes, cls):
        """
        Input:
            cluster_sizes: (Klass, Cluster=1, 3) [l,w,h], Klass is 0-based
            cls: (N), [klass], kclass is 1-based, 0-background
        Output
            mean_sizes: (N,3) [l,w,h]
        """
        """
        #TF version: (if N is not None)
        ##########
        N = cls.shape[0].value
        
        Ns = tf.reshape(tf.range(N), [N,1])

        K_mean_sizes = tf.reshape(cluster_sizes, [-1,3])
        K_mean_sizes = tf.concat([tf.constant([[1000.0, 1000.0, 1000.0]]), K_mean_sizes], axis=0)
        NK_mean_sizes = tf.tile(tf.expand_dims(K_mean_sizes, 0), [N,1,1])

        NK = tf.concat([Ns, tf.reshape(cls, [N,1])], axis=1) # (N,2)
        
        mean_sizes = tf.gather_nd(NK_mean_sizes, NK)
        return mean_sizes
        """

        # NumPy version: if N is None, by using tf.py_func, N should be determined
        #############
        K_mean_sizes = np.reshape(cluster_sizes, (-1, 3))
        K_mean_sizes = np.vstack(
            [np.asarray([1000.0, 1000.0, 1000.0]), K_mean_sizes]
        )  # insert 0-background
        mean_sizes = K_mean_sizes[cls]

        return mean_sizes.astype(np.float32)

    def build(self, **kwargs):
        self._set_up_input_pls()
        pc_pts = self.placeholders[self.PL_RPN_PTS]  # (B,P,3)
        pc_fts = self.placeholders[self.PL_RPN_FTS]  # (B,P,C)
        foreground_mask = self.placeholders[self.PL_RPN_FG_MASK]  # (B,P)
        pc_intensities = self.placeholders[self.PL_RPN_INTENSITY]  # (B,P)

        proposals = self.placeholders[self.PL_PROPOSALS]  # (B,n,7)
        proposals_iou3d = self.placeholders[self.PL_PROPOSALS_IOU]  # (B,n)
        proposals_gt_box3d = self.placeholders[self.PL_PROPOSALS_GT][
            :, :, :7
        ]  # (B,n,7)
        proposals_gt_cls = self.placeholders[self.PL_PROPOSALS_GT][:, :, 7]  # (B,n)

        """
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
        """
        # ROI Pooling
        with tf.variable_scope("avod_roi_pooling"):
            # Expand proposals' size
            with tf.variable_scope("expand_proposal"):
                expand_length = self._pooling_context_length
                expanded_size = proposals[:, :, 3:6] + 2 * expand_length
                expanded_proposals = tf.stack(
                    [
                        proposals[:, :, 0],
                        proposals[:, :, 1] + expand_length,
                        proposals[:, :, 2],
                        expanded_size[:, :, 0],
                        expanded_size[:, :, 1],
                        expanded_size[:, :, 2],
                        proposals[:, :, 6],
                    ],
                    axis=2,
                )  # (B,n,7)

            def get_box_indices(boxes):
                proposals_shape = boxes.get_shape().as_list()
                if any(dim is None for dim in proposals_shape):
                    proposals_shape = tf.shape(boxes)
                ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
                multiplier = tf.expand_dims(
                    tf.range(start=0, limit=proposals_shape[0]), 1
                )
                return tf.reshape(ones_mat * multiplier, [-1])

            tf_box_indices = get_box_indices(expanded_proposals)

            # Do ROI Pooling on PC
            proposals = tf.reshape(proposals, [-1, 7])  # (N=Bn,7)
            expanded_proposals = tf.reshape(expanded_proposals, [-1, 7])  # (N=Bn,7)
            proposals_iou3d = tf.reshape(proposals_iou3d, [-1])  # (N=Bn)
            proposals_gt_box3d = tf.reshape(proposals_gt_box3d, [-1, 7])  # (N=Bn,7)
            proposals_gt_cls = tf.reshape(proposals_gt_cls, [-1])  # (N=Bn)
            from cropping import tf_cropping

            crop_pts, crop_fts, crop_intensities, crop_mask, _, non_empty_box_mask = tf_cropping.pc_crop_and_sample(
                pc_pts,
                pc_fts,
                tf.expand_dims(pc_intensities, -1),
                foreground_mask,
                box_8c_encoder.tf_box_3d_to_box_8co(expanded_proposals),
                tf_box_indices,
                self._proposal_roi_crop_size,
            )  # (N,R,3), (N,R,C), (N,R,1) (N,R), _, (N)
            tf.summary.histogram(
                "non_empty_box_mask", tf.cast(non_empty_box_mask, tf.int8)
            )

            """
            # Do ROI Pooling on image
            img_rois = tf.image.crop_and_resize(
                img_feature_maps,
                img_proposal_boxes_norm_tf_order,
                tf_box_indices,
                self._proposal_roi_crop_size,
                name='img_rois')
            """
        with tf.variable_scope("local_spatial_feature"):
            with tf.variable_scope("canonical_transform"):
                crop_pts_ct = self._canonical_transform(crop_pts, proposals)

            with tf.variable_scope("distance_to_sensor"):
                crop_distance = (
                    tf.sqrt(
                        tf.square(crop_pts[:, :, 0])
                        + tf.square(crop_pts[:, :, 1])
                        + tf.square(crop_pts[:, :, 2])
                    )
                    / self._bev_extents[1, 1]
                    - 0.5
                )

            if self._use_intensity_feature:
                local_feature_input = tf.concat(
                    [
                        crop_pts_ct,
                        crop_intensities,
                        tf.expand_dims(tf.to_float(crop_mask), -1),
                        tf.expand_dims(crop_distance, -1),
                    ],
                    axis=-1,
                )
            else:
                local_feature_input = tf.concat(
                    [
                        crop_pts_ct,
                        tf.expand_dims(tf.to_float(crop_mask), -1),
                        tf.expand_dims(crop_distance, -1),
                    ],
                    axis=-1,
                )

            with tf.variable_scope("mlp"):
                fc_layers = [local_feature_input]
                layers_config = self._config.layers_config.avod_config.mlp
                for layer_idx, layer_param in enumerate(layers_config):
                    C = layer_param.C
                    dropout_rate = layer_param.dropout_rate
                    fc = pf.dense(
                        fc_layers[-1], C, "fc{:d}".format(layer_idx), self._is_training
                    )
                    fc_drop = tf.layers.dropout(
                        fc,
                        dropout_rate,
                        training=self._is_training,
                        name="fc{:d}_drop".format(layer_idx),
                    )
                    fc_layers.append(fc_drop)

        with tf.variable_scope("pc_encoder"):
            merged_fts = tf.concat([crop_fts, fc_layers[-1]], axis=-1)  # (N,R,2C)
            encode_pts, encode_fts = self._pc_feature_extractor.build(
                crop_pts, merged_fts, self._is_training
            )  # (N,r,3), (N,r,C')

        # branch-1: Box classification
        #########################################
        with tf.variable_scope("classification_confidence"):
            cls_multi_logits = pf.dense(
                encode_fts,
                self.num_classes + 1,
                "cls_multi_logits",
                self._is_training,
                with_bn=False,
                activation=None,
            )  # (N,r,K)
            cls_logits = tf.reduce_mean(
                cls_multi_logits, axis=1, name="cls_logits"
            )  # (N,K)
            cls_softmax = tf.nn.softmax(cls_logits, name="cls_softmax")  # (N,K)
            cls_preds = tf.argmax(cls_softmax, axis=-1, name="cls_predictions")
            cls_scores = tf.reduce_max(cls_softmax[:, 1:], axis=-1, name="cls_scores")

        # branch-2: bin-based 3D box refinement
        #########################################
        with tf.variable_scope("bin_based_box_refinement"):
            # Parse brn layers config
            encode_mean_fts = tf.reduce_mean(encode_fts, axis=1)  # (N,C')
            fc_layers = [encode_mean_fts]
            layers_config = self._config.layers_config.avod_config.fc_layer
            for layer_idx, layer_param in enumerate(layers_config):
                C = layer_param.C
                dropout_rate = layer_param.dropout_rate
                fc = pf.dense(
                    fc_layers[-1], C, "fc{:d}".format(layer_idx), self._is_training
                )
                fc_drop = tf.layers.dropout(
                    fc,
                    dropout_rate,
                    training=self._is_training,
                    name="fc{:d}_drop".format(layer_idx),
                )
                fc_layers.append(fc_drop)

            fc_output = pf.dense(
                fc_layers[-1],
                self.NUM_BIN_X * 2 + self.NUM_BIN_Z * 2 + self.NUM_BIN_THETA * 2 + 4,
                "fc_output",
                self._is_training,
                activation=None,
            )

        bin_x_logits, res_x_norms, bin_z_logits, res_z_norms, bin_theta_logits, res_theta_norms, res_y, res_size_norm = self._parse_brn_output(
            fc_output
        )
        res_y = tf.squeeze(res_y, [-1])

        # Final Predictions
        ######################################################
        with tf.variable_scope("boxes"):
            bin_x = tf.argmax(bin_x_logits, axis=-1)  # (N)
            bin_z = tf.argmax(bin_z_logits, axis=-1)  # (N)
            bin_theta = tf.argmax(bin_theta_logits, axis=-1)  # (N)

            res_x_norm, res_z_norm, res_theta_norm = tf.py_func(
                self._gather_residuals,
                [res_x_norms, res_z_norms, res_theta_norms, bin_x, bin_z, bin_theta],
                (tf.float32, tf.float32, tf.float32),
            )

            # NMS
            if self._train_val_test == "train":
                # to speed up training, skip NMS, as we don't care what top_* is during training
                print("Skip BRN-NMS during training")
            else:
                mean_sizes = tf.py_func(
                    self._gather_mean_sizes,
                    [tf.convert_to_tensor(np.asarray(self._cluster_sizes)), cls_preds],
                    tf.float32,
                )
                # Decode bin-based 3D Box
                with tf.variable_scope("decoding"):
                    reg_boxes_3d = bin_based_box3d_encoder.tf_decode(
                        proposals[:, :3],
                        proposals[:, 6],
                        bin_x,
                        res_x_norm,
                        bin_z,
                        res_z_norm,
                        bin_theta,
                        res_theta_norm,
                        res_y,
                        res_size_norm,
                        mean_sizes,
                        self.S,
                        self.DELTA,
                        self.R,
                        self.DELTA_THETA,
                    )  # (N,7)

                # oriented-NMS is much slower than non-oriented-NMS (tf.image.non_max_suppression)
                # while get significant higher proposal recall@IoU=0.7
                oriented_NMS = True
                print("RCNN oriented_NMS = " + str(oriented_NMS))
                # bev-NMS and ignore multiclass
                with tf.variable_scope("bev_nms"):

                    def sb_nms_fn(args):
                        (sb_boxes, sb_scores, sb_non_empty_box_mask) = args
                        sb_boxes = tf.boolean_mask(sb_boxes, sb_non_empty_box_mask)
                        sb_scores = tf.boolean_mask(sb_scores, sb_non_empty_box_mask)
                        if oriented_NMS:
                            sb_nms_indices = compute_iou.oriented_nms_tf(
                                sb_boxes, sb_scores, self._nms_iou_thresh
                            )
                            sb_nms_indices = sb_nms_indices[
                                : tf.minimum(
                                    self._nms_size, tf.shape(sb_nms_indices)[0]
                                )
                            ]
                        else:
                            # ortho rotating
                            sb_box_anchors = box_3d_encoder.tf_box_3d_to_anchor(
                                sb_boxes
                            )
                            sb_bev_boxes, _ = anchor_projector.project_to_bev(
                                sb_box_anchors, self._bev_extents
                            )
                            sb_bev_boxes_tf_order = anchor_projector.reorder_projected_boxes(
                                sb_bev_boxes
                            )
                            sb_nms_indices = tf.image.non_max_suppression(
                                sb_bev_boxes_tf_order,
                                sb_scores,
                                max_output_size=self._nms_size,
                                iou_threshold=self._nms_iou_thresh,
                            )

                        sb_nms_indices = tf.cond(
                            tf.greater(self._nms_size, tf.shape(sb_nms_indices)[0]),
                            true_fn=lambda: tf.pad(
                                sb_nms_indices,
                                [[0, self._nms_size - tf.shape(sb_nms_indices)[0]]],
                                mode="CONSTANT",
                                constant_values=-1,
                            ),
                            false_fn=lambda: sb_nms_indices,
                        )

                        return sb_nms_indices, tf.shape(sb_nms_indices)[0]

                    batch_reg_boxes_3d = tf.reshape(
                        reg_boxes_3d, [self._batch_size, -1, 7]
                    )  # (B,n,7)
                    batch_cls_scores = tf.reshape(
                        cls_scores, [self._batch_size, -1]
                    )  # (B,n)
                    batch_cls_softmax = tf.reshape(
                        cls_softmax, [self._batch_size, -1, self.num_classes + 1]
                    )  # (B,n,K)
                    batch_non_empty_box_mask = tf.reshape(
                        non_empty_box_mask, [self._batch_size, -1]
                    )  # (B,n)

                    nms_indices, num_proposals_before_padding = tf.map_fn(
                        sb_nms_fn,
                        elems=[
                            batch_reg_boxes_3d,
                            batch_cls_scores,
                            batch_non_empty_box_mask,
                        ],
                        dtype=(tf.int32, tf.int32),
                    )

        ######################################################
        # Determine Positive/Negative GTs for the loss function & metrics
        ######################################################
        # for box cls loss
        with tf.variable_scope("box_cls_gt"):
            neg_cls_mask = tf.less(proposals_iou3d, self.dataset.cls_neg_iou_range[1])
            pos_cls_mask = tf.greater(
                proposals_iou3d, self.dataset.cls_pos_iou_range[0]
            )
            pos_neg_cls_mask = tf.logical_or(neg_cls_mask, pos_cls_mask)
            pos_neg_cls_mask = tf.logical_and(pos_neg_cls_mask, non_empty_box_mask)

            # cls gt
            cls_gt = tf.where(
                neg_cls_mask, tf.zeros_like(proposals_gt_cls), proposals_gt_cls
            )
            cls_gt_one_hot = tf.one_hot(
                tf.to_int32(cls_gt),
                depth=self.num_classes + 1,
                on_value=1.0,
                off_value=0.0,
            )

        # for box refinement loss
        with tf.variable_scope("box_cls_reg_gt"):
            pos_reg_mask = tf.greater(
                proposals_iou3d, self.dataset.reg_pos_iou_range[0]
            )
            pos_reg_mask = tf.logical_and(pos_reg_mask, non_empty_box_mask)

            mean_sizes = tf.py_func(
                self._gather_mean_sizes,
                [
                    tf.convert_to_tensor(np.asarray(self._cluster_sizes)),
                    tf.to_int32(proposals_gt_cls),
                ],
                tf.float32,
            )
            # reg gt
            (
                bin_x_gt,
                res_x_gt,
                bin_z_gt,
                res_z_gt,
                bin_theta_gt,
                res_theta_gt,
                res_y_gt,
                res_size_norm_gt,
            ) = bin_based_box3d_encoder.tf_encode(
                proposals[:, :3],
                proposals[:, 6],
                proposals_gt_box3d,
                mean_sizes,
                self.S,
                self.DELTA,
                self.R,
                self.DELTA_THETA,
            )

            bin_x_gt_one_hot = tf.one_hot(
                tf.to_int32(bin_x_gt),
                depth=int(2 * self.S / self.DELTA),
                on_value=1.0,
                off_value=0.0,
            )

            bin_z_gt_one_hot = tf.one_hot(
                tf.to_int32(bin_z_gt),
                depth=int(2 * self.S / self.DELTA),
                on_value=1.0,
                off_value=0.0,
            )

            bin_theta_gt_one_hot = tf.one_hot(
                tf.to_int32(bin_theta_gt),
                depth=int(2 * self.R / self.DELTA_THETA),
                on_value=1.0,
                off_value=0.0,
            )

        ######################################################
        # Prediction Dict
        ######################################################
        prediction_dict = dict()
        if self._train_val_test in ["train", "val"]:
            # cls Mini batch preds & gt
            prediction_dict[self.PRED_MB_CLASSIFICATION_LOGITS] = cls_logits
            prediction_dict[self.PRED_MB_CLASSIFICATIONS_GT] = cls_gt_one_hot
            prediction_dict[self.PRED_MB_CLASSIFICATION_MASK] = pos_neg_cls_mask

            # reg Mini batch preds
            prediction_dict[self.PRED_MB_CLS] = (
                bin_x_logits,
                bin_z_logits,
                bin_theta_logits,
            )
            prediction_dict[self.PRED_MB_REG] = (
                res_x_norm,
                res_z_norm,
                res_theta_norm,
                res_y,
                res_size_norm,
            )

            # reg Mini batch gt
            prediction_dict[self.PRED_MB_CLS_GT] = (
                bin_x_gt_one_hot,
                bin_z_gt_one_hot,
                bin_theta_gt_one_hot,
            )
            prediction_dict[self.PRED_MB_REG_GT] = (
                res_x_gt,
                res_z_gt,
                res_theta_gt,
                res_y_gt,
                res_size_norm_gt,
            )

            # reg Mini batch pos mask
            prediction_dict[self.PRED_MB_POS_REG_MASK] = pos_reg_mask

            if self._train_val_test == "val":
                prediction_dict[self.PRED_BOXES] = batch_reg_boxes_3d
                prediction_dict[self.PRED_SOFTMAX] = batch_cls_softmax
                prediction_dict[self.PRED_NON_EMPTY_BOX_MASK] = batch_non_empty_box_mask
                prediction_dict[self.PRED_NMS_INDICES] = nms_indices
        else:
            prediction_dict[self.PRED_BOXES] = batch_reg_boxes_3d
            prediction_dict[self.PRED_SOFTMAX] = batch_cls_softmax
            prediction_dict[self.PRED_NON_EMPTY_BOX_MASK] = batch_non_empty_box_mask
            prediction_dict[self.PRED_NMS_INDICES] = nms_indices
        return prediction_dict

    def _parse_brn_output(self, brn_output):
        """
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

            res_size_norm: (l,w,h)
        """
        bin_x_logits = tf.slice(brn_output, [0, 0], [-1, self.NUM_BIN_X])
        res_x_norms = tf.slice(brn_output, [0, self.NUM_BIN_X], [-1, self.NUM_BIN_X])

        bin_z_logits = tf.slice(
            brn_output, [0, self.NUM_BIN_X * 2], [-1, self.NUM_BIN_Z]
        )
        res_z_norms = tf.slice(
            brn_output, [0, self.NUM_BIN_X * 2 + self.NUM_BIN_Z], [-1, self.NUM_BIN_Z]
        )

        bin_theta_logits = tf.slice(
            brn_output,
            [0, self.NUM_BIN_X * 2 + self.NUM_BIN_Z * 2],
            [-1, self.NUM_BIN_THETA],
        )
        res_theta_norms = tf.slice(
            brn_output,
            [0, self.NUM_BIN_X * 2 + self.NUM_BIN_Z * 2 + self.NUM_BIN_THETA],
            [-1, self.NUM_BIN_THETA],
        )

        res_y = tf.slice(
            brn_output,
            [0, self.NUM_BIN_X * 2 + self.NUM_BIN_Z * 2 + self.NUM_BIN_THETA * 2],
            [-1, 1],
        )

        res_size_norm = tf.slice(
            brn_output,
            [0, self.NUM_BIN_X * 2 + self.NUM_BIN_Z * 2 + self.NUM_BIN_THETA * 2 + 1],
            [-1, 3],
        )

        return (
            bin_x_logits,
            res_x_norms,
            bin_z_logits,
            res_z_norms,
            bin_theta_logits,
            res_theta_norms,
            res_y,
            res_size_norm,
        )

    def create_feed_dict(self, batch_size=1, sample_index=None):

        if batch_size != self._batch_size:
            raise ValueError("feed batch_size must equal to model build batch_size")

        if self._train_val_test in ["train", "val"]:

            # sample_index should be None
            if sample_index is not None:
                raise ValueError(
                    "sample_index should be None. Do not load "
                    "particular samples during train or val"
                )

            if self._train_val_test == "train":
                # Get the a random sample from the remaining epoch
                batch_data, sample_names = self.dataset.next_batch(
                    batch_size, True, model="rcnn"
                )

            else:  # self._train_val_test == "val"
                # Load samples in order for validation
                batch_data, sample_names = self.dataset.next_batch(
                    batch_size, False, model="rcnn"
                )
        else:
            # For testing, any sample should work
            if sample_index is not None:
                samples = self.dataset.load_samples([sample_index], model="rcnn")
                batch_data, sample_names = self.dataset.collate_batch(samples)
            else:
                batch_data, sample_names = self.dataset.next_batch(
                    batch_size, False, model="rcnn"
                )

        self._placeholder_inputs[self.PL_PROPOSALS] = batch_data[constants.KEY_RPN_ROI]
        self._placeholder_inputs[self.PL_PROPOSALS_IOU] = batch_data[
            constants.KEY_RPN_IOU
        ]
        self._placeholder_inputs[self.PL_PROPOSALS_GT] = batch_data[
            constants.KEY_RPN_GT
        ]

        self._placeholder_inputs[self.PL_RPN_PTS] = batch_data[constants.KEY_RPN_PTS]
        self._placeholder_inputs[self.PL_RPN_INTENSITY] = batch_data[
            constants.KEY_RPN_INTENSITY
        ]
        self._placeholder_inputs[self.PL_RPN_FG_MASK] = batch_data[
            constants.KEY_RPN_FG_MASK
        ].astype(np.bool)
        self._placeholder_inputs[self.PL_RPN_FTS] = batch_data[constants.KEY_RPN_FTS]

        self._sample_names = sample_names

        feed_dict = dict()
        for key, value in self.placeholders.items():
            feed_dict[value] = self._placeholder_inputs[key]
        return feed_dict

    def loss(self, prediction_dict):
        # cls Mini batch preds & gt
        cls_logits = prediction_dict[self.PRED_MB_CLASSIFICATION_LOGITS]
        cls_gt_one_hot = prediction_dict[self.PRED_MB_CLASSIFICATIONS_GT]

        with tf.variable_scope("brn_losses"):
            with tf.variable_scope("box_classification"):
                pos_neg_cls_mask = prediction_dict[self.PRED_MB_CLASSIFICATION_MASK]
                cls_loss = losses.WeightedSoftmaxLoss()
                cls_loss_weight = self._config.loss_config.cls_loss_weight
                box_classification_loss = cls_loss(
                    cls_logits,
                    cls_gt_one_hot,
                    weight=cls_loss_weight,
                    mask=pos_neg_cls_mask,
                )

                with tf.variable_scope("cls_norm"):
                    # normalize by the number of boxes
                    num_cls_boxes = tf.reduce_sum(tf.cast(pos_neg_cls_mask, tf.float32))
                    # with tf.control_dependencies(
                    #    [tf.assert_positive(num_cls_boxes)]):
                    #    box_classification_loss /= num_cls_boxes
                    box_classification_loss = tf.cond(
                        tf.greater(num_cls_boxes, 0),
                        true_fn=lambda: box_classification_loss / num_cls_boxes,
                        false_fn=lambda: box_classification_loss,
                    )
                    tf.summary.scalar("box_classification", box_classification_loss)

            # these should include positive boxes only
            with tf.variable_scope("bin_classification"):
                # reg Mini batch pos mask
                pos_reg_mask = prediction_dict[self.PRED_MB_POS_REG_MASK]
                bin_classification_loss = 0.0
                # bin_x_logits, bin_z_logits, bin_theta_logits = prediction_dict[self.PRED_MB_CLS]
                # bin_x_gt_one_hot, bin_z_gt_one_hot, bin_theta_gt_one_hot = prediction_dict[self.PRED_MB_CLS_GT]
                for elem in zip(
                    prediction_dict[self.PRED_MB_CLS],
                    prediction_dict[self.PRED_MB_CLS_GT],
                ):
                    bin_classification_loss += cls_loss(
                        elem[0], elem[1], weight=cls_loss_weight, mask=pos_reg_mask
                    )
                with tf.variable_scope("cls_norm"):
                    # normalize by the number of positive boxes
                    num_reg_boxes = tf.reduce_sum(tf.cast(pos_reg_mask, tf.float32))
                    # with tf.control_dependencies(
                    #    [tf.assert_positive(num_reg_boxes)]):
                    #    bin_classification_loss /= num_reg_boxes
                    bin_classification_loss = tf.cond(
                        tf.greater(num_reg_boxes, 0),
                        true_fn=lambda: bin_classification_loss / num_reg_boxes,
                        false_fn=lambda: bin_classification_loss,
                    )
                    tf.summary.scalar("bin_classification", bin_classification_loss)

            # these should include positive boxes only
            with tf.variable_scope("regression"):
                reg_loss = losses.WeightedSmoothL1Loss()
                reg_loss_weight = self._config.loss_config.reg_loss_weight
                regression_loss = 0.0
                # res_x_norm, res_z_norm, res_theta_norm, res_y, res_size_norm = prediction_dict[self.PRED_MB_REG]
                # res_x_gt, res_z_gt, res_theta_gt, res_y_gt, res_size_norm_gt = prediction_dict[self.PRED_MB_REG_GT]
                for elem in zip(
                    prediction_dict[self.PRED_MB_REG],
                    prediction_dict[self.PRED_MB_REG_GT],
                ):
                    regression_loss += reg_loss(
                        elem[0], elem[1], weight=reg_loss_weight, mask=pos_reg_mask
                    )
                with tf.variable_scope("reg_norm"):
                    # normalize by the number of positive boxes
                    # with tf.control_dependencies(
                    #    [tf.assert_positive(num_reg_boxes)]):
                    #    regression_loss /= num_reg_boxes
                    regression_loss = tf.cond(
                        tf.greater(num_reg_boxes, 0),
                        true_fn=lambda: regression_loss / num_reg_boxes,
                        false_fn=lambda: regression_loss,
                    )
                    tf.summary.scalar("regression", regression_loss)

            with tf.variable_scope("rcnn_loss"):
                rcnn_loss = (
                    box_classification_loss + bin_classification_loss + regression_loss
                )

        loss_dict = {
            self.LOSS_FINAL_CLASSIFICATION: box_classification_loss,
            self.LOSS_FINAL_BIN_CLASSIFICATION: bin_classification_loss,
            self.LOSS_FINAL_REGRESSION: regression_loss,
        }

        return loss_dict, rcnn_loss
