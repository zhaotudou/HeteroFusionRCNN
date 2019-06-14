import numpy as np

import tensorflow as tf

from avod.builders import feature_extractor_builder
from avod.core import anchor_projector
from avod.core import box_3d_encoder
from avod.core import box_8c_encoder
from avod.core import projection
from avod.core import bin_based_box3d_encoder
from avod.core import pointfly as pf
from avod.core import compute_iou

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

    PL_IMG_INPUT = "img_input_pl"
    PL_CALIB_P2 = "frame_calib_p2"
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
    PRED_NUM_BOXES_BEFORE_PADDING = "avod_prediction_num_boxes_before_padding"

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

        self._img_h = input_config.img_dims_h
        self._img_w = input_config.img_dims_w
        self._img_depth = input_config.img_depth

        # AVOD config
        avod_config = self._config.avod_config
        self._use_intensity_feature = avod_config.avod_use_intensity_feature
        self._fusion_method = avod_config.avod_fusion_method
        self._proposal_roi_crop_size = avod_config.avod_proposal_roi_crop_size
        self._proposal_roi_img_crop_size = [
            avod_config.avod_proposal_roi_img_crop_size
        ] * 2
        self._nms_size = avod_config.avod_nms_size
        self._nms_iou_thresh = avod_config.avod_nms_iou_thresh
        self._path_drop_probabilities = self._config.path_drop_probabilities

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
        self._img_feature_extractor = feature_extractor_builder.get_extractor(
            self._config.layers_config.avod_config.img_feature_extractor
        )

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
                [self._batch_size, self._pc_sample_pts, 256 + 32],
                self.PL_RPN_FTS,
            )

        with tf.variable_scope("img_input"):
            img_input_placeholder = self._add_placeholder(
                tf.float32,
                [self._batch_size, self._img_h, self._img_w, self._img_depth],
                self.PL_IMG_INPUT,
            )

            self._img_preprocessed = self._img_feature_extractor.preprocess_input(
                img_input_placeholder
            )

        with tf.variable_scope("sample_info"):
            # the calib matrix shape is (3 x 4)
            self._add_placeholder(
                tf.float32, [self._batch_size, 3, 4], self.PL_CALIB_P2
            )

    def _set_up_feature_extractors(self):
        self._img_fts, _ = self._img_feature_extractor.build(
            self._img_preprocessed, self._is_training
        )  # (B,H,W,C1)

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
        """
        N = tf.shape(bin_x)[0]
        Ns = tf.reshape(tf.range(N), [N, 1])

        NK_x = tf.concat([Ns, tf.reshape(bin_x, [N, 1])], axis=1)  # (N,2)
        res_x_norm = tf.gather_nd(res_x_norms, NK_x)  # (N)

        NK_z = tf.concat([Ns, tf.reshape(bin_z, [N, 1])], axis=1)  # (N,2)
        res_z_norm = tf.gather_nd(res_z_norms, NK_z)  # (N)

        NK_theta = tf.concat([Ns, tf.reshape(bin_theta, [N, 1])], axis=1)  # (N,2)
        res_theta_norm = tf.gather_nd(res_theta_norms, NK_theta)  # (N)

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
        """

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
        """
        N = tf.shape(cls)[0]

        Ns = tf.reshape(tf.range(N), [N, 1])

        K_mean_sizes = tf.reshape(cluster_sizes, [-1, 3])
        # insert 0-background mean size as mean size of all foreground class
        K_mean_sizes = tf.concat(
            [tf.expand_dims(tf.reduce_mean(K_mean_sizes, 0), axis=0), K_mean_sizes],
            axis=0,
        )
        NK_mean_sizes = tf.tile(tf.expand_dims(K_mean_sizes, 0), [N, 1, 1])

        NK = tf.concat([Ns, tf.reshape(cls, [N, 1])], axis=1)  # (N,2)

        mean_sizes = tf.gather_nd(NK_mean_sizes, NK)
        return mean_sizes

        """
        # NumPy version: if N is None, by using tf.py_func, N should be determined
        #############
        K_mean_sizes = np.reshape(cluster_sizes, (-1, 3))
        K_mean_sizes = np.vstack(
            [np.mean(K_mean_sizes, axis=0), K_mean_sizes]
        )  # insert 0-background
        mean_sizes = K_mean_sizes[cls]

        return mean_sizes.astype(np.float32)
        """

    def build(self, **kwargs):
        self._set_up_input_pls()
        self._set_up_feature_extractors()

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

        # ROI Pooling
        with tf.variable_scope("avod_roi_pooling"):

            def get_box_indices(boxes):
                proposals_shape = boxes.get_shape().as_list()
                if any(dim is None for dim in proposals_shape):
                    proposals_shape = tf.shape(boxes)
                ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
                multiplier = tf.expand_dims(
                    tf.range(start=0, limit=proposals_shape[0]), 1
                )
                return tf.reshape(ones_mat * multiplier, [-1])

            tf_box_indices = get_box_indices(proposals)

            def sb_project_to_image_space(args):
                (sb_proposal, sb_calib, sb_image_shape) = args
                return projection.tf_project_to_image_space(
                    sb_proposal, sb_calib, sb_image_shape
                )

            _, proj_proposals_box2d_norm = tf.map_fn(
                sb_project_to_image_space,
                elems=[
                    proposals,
                    self.placeholders[self.PL_CALIB_P2],
                    tf.tile(
                        tf.expand_dims(tf.constant([self._img_h, self._img_w]), axis=0),
                        [self._batch_size, 1],
                    ),
                ],
                dtype=(tf.float32, tf.float32),
            )  # (B,n,4)
            # y1, x1, y2, x2
            proj_proposals_box2d_norm_reorder = anchor_projector.reorder_projected_boxes(
                tf.reshape(proj_proposals_box2d_norm, [-1, 4])
            )  # (N=Bn,4)

            proposals = tf.reshape(proposals, [-1, 7])  # (N=Bn,7)
            proposals_iou3d = tf.reshape(proposals_iou3d, [-1])  # (N=Bn)
            proposals_gt_box3d = tf.reshape(proposals_gt_box3d, [-1, 7])  # (N=Bn,7)
            proposals_gt_cls = tf.reshape(proposals_gt_cls, [-1])  # (N=Bn)

            # Expand proposals' size
            with tf.variable_scope("expand_proposal"):
                expand_length = self._pooling_context_length
                expanded_size = proposals[:, 3:6] + 2 * expand_length
                expanded_proposals = tf.stack(
                    [
                        proposals[:, 0],
                        proposals[:, 1] + expand_length,
                        proposals[:, 2],
                        expanded_size[:, 0],
                        expanded_size[:, 1],
                        expanded_size[:, 2],
                        proposals[:, 6],
                    ],
                    axis=1,
                )  # (N=Bn,7)

            from cropping import tf_cropping

            # Do ROI Pooling on PC
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

            # Do ROI Pooling on image
            img_rois = tf.image.crop_and_resize(
                self._img_fts,
                proj_proposals_box2d_norm_reorder,
                tf_box_indices,
                self._proposal_roi_img_crop_size,
                name="img_rois",
            )  # (N,r1,r1,C1)

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
            _, pc_rois = self._pc_feature_extractor.build(
                crop_pts_ct, merged_fts, self._is_training
            )  # (N,r,C')

        # fuse roi pc + img features
        #########################################
        with tf.variable_scope("fts_fuse"):
            fusion_mean_div_factor = 2.0
            if not (
                self._path_drop_probabilities[0]
                == self._path_drop_probabilities[1]
                == 1.0
            ):

                with tf.variable_scope("avod_path_drop"):
                    random_values = tf.random_uniform(shape=[3], minval=0.0, maxval=1.0)
                    img_mask, pc_mask = self.create_path_drop_masks(
                        self._path_drop_probabilities[0],
                        self._path_drop_probabilities[1],
                        random_values,
                    )
                    pc_rois = tf.multiply(pc_rois, pc_mask)  # (N,r,C')
                    img_rois = tf.multiply(img_rois, img_mask)  # (N,r1,r1,C1)

                    # Overwrite the division factor
                    fusion_mean_div_factor = img_mask + pc_mask

            if self._fusion_method == "mean_concat":
                pc_rois = tf.reduce_mean(pc_rois, axis=1)  # (N,C')
                img_rois = tf.reduce_mean(img_rois, axis=1)
                img_rois = tf.reduce_mean(img_rois, axis=1)  # (N,C1)
                fuse_rois = tf.concat([pc_rois, img_rois], axis=-1)  # (N,C'+C1)
            elif self._fusion_method == "flat_concat":
                pc_rois = tf.layers.flatten(pc_rois)  # (N, rC')
                img_rois = tf.layers.flatten(img_rois)  # (N, r1r1C1)
                fuse_rois = tf.concat([pc_rois, img_rois], axis=-1)  # (N,rC'+r1r1C1)
            else:
                raise ValueError("Invalid fusion method", self._fusion_method)

        # branch-1: Box classification
        #########################################
        with tf.variable_scope("classification_confidence"):
            # Parse brn layers config
            fc_layers = [fuse_rois]
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

            cls_logits = pf.dense(
                fc_layers[-1],
                self.num_classes + 1,
                "cls_logits",
                self._is_training,
                with_bn=False,
                activation=None,
            )  # (N,K)
            cls_softmax = tf.nn.softmax(cls_logits, name="cls_softmax")  # (N,K)
            cls_preds = tf.argmax(
                cls_softmax, axis=-1, name="cls_predictions", output_type=tf.int32
            )
            cls_scores = tf.reduce_max(cls_softmax[:, 1:], axis=-1, name="cls_scores")

        # branch-2: bin-based 3D box refinement
        #########################################
        with tf.variable_scope("bin_based_box_refinement"):
            # Parse brn layers config
            fc_layers = [fuse_rois]
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
                "reg_output",
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
            # NMS
            if self._train_val_test == "train":
                # to speed up training, skip NMS, as we don't care what top_* is during training
                print("Skip BRN-NMS during training")
            else:
                # Decode bin-based 3D Box
                with tf.variable_scope("decoding"):
                    bin_x = tf.argmax(
                        bin_x_logits, axis=-1, output_type=tf.int32
                    )  # (N)
                    bin_z = tf.argmax(
                        bin_z_logits, axis=-1, output_type=tf.int32
                    )  # (N)
                    bin_theta = tf.argmax(
                        bin_theta_logits, axis=-1, output_type=tf.int32
                    )  # (N)

                    res_x_norm, res_z_norm, res_theta_norm = self._gather_residuals(
                        res_x_norms,
                        res_z_norms,
                        res_theta_norms,
                        bin_x,
                        bin_z,
                        bin_theta,
                    )

                    mean_sizes = self._gather_mean_sizes(
                        tf.convert_to_tensor(
                            np.asarray(self._cluster_sizes, dtype=np.float32)
                        ),
                        cls_preds,
                    )
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

                        sb_nms_indices_padded = tf.cond(
                            tf.greater(self._nms_size, tf.shape(sb_nms_indices)[0]),
                            true_fn=lambda: tf.pad(
                                sb_nms_indices,
                                [[0, self._nms_size - tf.shape(sb_nms_indices)[0]]],
                                mode="CONSTANT",
                                constant_values=-1,
                            ),
                            false_fn=lambda: sb_nms_indices,
                        )

                        return sb_nms_indices_padded, tf.shape(sb_nms_indices)[0]

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

                    nms_indices, num_boxes_before_padding = tf.map_fn(
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

            mean_sizes = self._gather_mean_sizes(
                tf.convert_to_tensor(np.asarray(self._cluster_sizes, dtype=np.float32)),
                tf.to_int32(proposals_gt_cls),
            )
            # reg gt
            (
                bin_x_gt,
                res_x_norm_gt,
                bin_z_gt,
                res_z_norm_gt,
                bin_theta_gt,
                res_theta_norm_gt,
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

            res_x_norm, res_z_norm, res_theta_norm = self._gather_residuals(
                res_x_norms,
                res_z_norms,
                res_theta_norms,
                bin_x_gt,
                bin_z_gt,
                bin_theta_gt,
            )

            bin_x_gt_one_hot = tf.one_hot(
                bin_x_gt,
                depth=int(2 * self.S / self.DELTA),
                on_value=1.0,
                off_value=0.0,
            )

            bin_z_gt_one_hot = tf.one_hot(
                bin_z_gt,
                depth=int(2 * self.S / self.DELTA),
                on_value=1.0,
                off_value=0.0,
            )

            bin_theta_gt_one_hot = tf.one_hot(
                bin_theta_gt,
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
                res_x_norm_gt,
                res_z_norm_gt,
                res_theta_norm_gt,
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
                prediction_dict[
                    self.PRED_NUM_BOXES_BEFORE_PADDING
                ] = num_boxes_before_padding
        else:
            prediction_dict[self.PRED_BOXES] = batch_reg_boxes_3d
            prediction_dict[self.PRED_SOFTMAX] = batch_cls_softmax
            prediction_dict[self.PRED_NON_EMPTY_BOX_MASK] = batch_non_empty_box_mask
            prediction_dict[self.PRED_NMS_INDICES] = nms_indices
            prediction_dict[
                self.PRED_NUM_BOXES_BEFORE_PADDING
            ] = num_boxes_before_padding

            output_reg_boxes_3d = tf.identity(
                batch_reg_boxes_3d, name="output_reg_boxes_3d"
            )
            output_cls_softmax = tf.identity(
                batch_cls_softmax, name="output_cls_softmax"
            )
            output_non_empty_box_mask = tf.identity(
                batch_non_empty_box_mask, name="output_non_empty_box_mask"
            )
            output_nms_indices = tf.identity(nms_indices, name="output_nms_indices")
            output_num_boxes_before_padding = tf.identity(
                num_boxes_before_padding, name="output_num_boxes_before_padding"
            )

            if self._batch_size == 1:
                self._batch_prediction_to_final_prediction(
                    batch_reg_boxes_3d,
                    batch_cls_softmax,
                    batch_non_empty_box_mask,
                    nms_indices,
                    num_boxes_before_padding,
                )

        return prediction_dict

    def _batch_prediction_to_final_prediction(
        self,
        batch_boxes,
        batch_softmax,
        batch_non_empty_mask,
        batch_nms_indices,
        batch_num_boxes,
    ):
        """Convert batch prediction to final prediction with 3d box, class score and class.
           This function could be called only when batch size is exactly 1.
           Note that final predicted boxes may have duplicated boxes. """
        boxes = tf.squeeze(batch_boxes)
        softmax = tf.squeeze(batch_softmax)
        num_boxes_before_padding = tf.squeeze(batch_num_boxes)
        nms_indices = tf.squeeze(batch_nms_indices)
        non_empty_box_mask = tf.squeeze(batch_non_empty_mask)

        # exclude empty boxes and duplicated boxes incurred by nms padding
        non_empty_box_indices = tf.where(non_empty_box_mask)
        non_empty_boxes = tf.gather(boxes, non_empty_box_indices)
        non_empty_softmax = tf.gather(softmax, non_empty_box_indices)
        final_boxes = tf.gather(non_empty_boxes, nms_indices[:num_boxes_before_padding])
        final_boxes = tf.squeeze(final_boxes, name="final_boxes")
        final_pred_softmax = tf.gather(
            non_empty_softmax, nms_indices[:num_boxes_before_padding]
        )
        final_pred_softmax = tf.squeeze(final_pred_softmax)

        # get box's class and score
        not_bkg_scores = tf.slice(final_pred_softmax, [0, 1], [-1, -1])
        final_pred_types = tf.argmax(
            not_bkg_scores, axis=1, output_type=tf.int32, name="final_box_classes"
        )
        box_range = tf.range(tf.shape(final_boxes)[0], dtype=tf.int32)
        final_pred_scores_indices = tf.stack([box_range, final_pred_types], axis=1)
        final_pred_scores = tf.gather_nd(
            not_bkg_scores, final_pred_scores_indices, name="final_box_class_scores"
        )

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
                    batch_size, True, model="rcnn", img_w=self._img_w, img_h=self._img_h
                )

            else:  # self._train_val_test == "val"
                # Load samples in order for validation
                batch_data, sample_names = self.dataset.next_batch(
                    batch_size,
                    False,
                    model="rcnn",
                    img_w=self._img_w,
                    img_h=self._img_h,
                )
        else:
            # For testing, any sample should work
            if sample_index is not None:
                samples = self.dataset.load_samples(
                    [sample_index], model="rcnn", img_w=self._img_w, img_h=self._img_h
                )
                batch_data, sample_names = self.dataset.collate_batch(samples)
            else:
                batch_data, sample_names = self.dataset.next_batch(
                    batch_size,
                    False,
                    model="rcnn",
                    img_w=self._img_w,
                    img_h=self._img_h,
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

        self._placeholder_inputs[self.PL_IMG_INPUT] = batch_data[
            constants.KEY_IMAGE_INPUT
        ]
        self._placeholder_inputs[self.PL_CALIB_P2] = batch_data[
            constants.KEY_STEREO_CALIB_P2
        ]
        self._sample_names = sample_names

        feed_dict = dict()
        for key, value in self.placeholders.items():
            feed_dict[value] = self._placeholder_inputs[key]
        return feed_dict

    def loss(self, prediction_dict):
        with tf.variable_scope("brn_losses"):
            with tf.variable_scope("box_classification"):
                cls_logits = prediction_dict[self.PRED_MB_CLASSIFICATION_LOGITS]
                cls_gt_one_hot = prediction_dict[self.PRED_MB_CLASSIFICATIONS_GT]
                pos_neg_cls_mask = prediction_dict[self.PRED_MB_CLASSIFICATION_MASK]
                masked_cls_logits = tf.boolean_mask(cls_logits, pos_neg_cls_mask)
                masked_cls_gt_one_hot = tf.boolean_mask(
                    cls_gt_one_hot, pos_neg_cls_mask
                )

                cls_loss = losses.WeightedSoftmaxLoss()
                cls_loss_weight = self._config.loss_config.cls_loss_weight
                box_classification_loss = cls_loss(
                    masked_cls_logits, masked_cls_gt_one_hot, weight=cls_loss_weight
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
                        false_fn=lambda: box_classification_loss * 0.0,
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
                    masked_pred_cls = tf.boolean_mask(elem[0], pos_reg_mask)
                    masked_pred_cls_gt = tf.boolean_mask(elem[1], pos_reg_mask)
                    bin_classification_loss += cls_loss(
                        masked_pred_cls, masked_pred_cls_gt, weight=cls_loss_weight
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
                        false_fn=lambda: bin_classification_loss * 0.0,
                    )
                    tf.summary.scalar("bin_classification", bin_classification_loss)

            # these should include positive boxes only
            with tf.variable_scope("regression"):
                reg_loss = losses.WeightedSmoothL1Loss()
                reg_loss_weight = self._config.loss_config.reg_loss_weight
                regression_loss = 0.0
                # res_x_norm, res_z_norm, res_theta_norm, res_y, res_size_norm = prediction_dict[self.PRED_MB_REG]
                # res_x_norm_gt, res_z_norm_gt, res_theta_norm_gt, res_y_gt, res_size_norm_gt = prediction_dict[self.PRED_MB_REG_GT]
                for idx, elem in enumerate(
                    zip(
                        prediction_dict[self.PRED_MB_REG],
                        prediction_dict[self.PRED_MB_REG_GT],
                    )
                ):
                    masked_pred_reg = tf.boolean_mask(elem[0], pos_reg_mask)
                    masked_pred_reg_gt = tf.boolean_mask(elem[1], pos_reg_mask)
                    elem_reg_loss = reg_loss(
                        masked_pred_reg, masked_pred_reg_gt, weight=reg_loss_weight
                    )
                    regression_loss += elem_reg_loss
                    """
                    with tf.variable_scope("elem_reg_norm_" + str(idx)):
                        # normalize by the number of positive boxes
                        # with tf.control_dependencies(
                        #    [tf.assert_positive(num_reg_boxes)]):
                        #    regression_loss /= num_reg_boxes
                        elem_reg_loss = tf.cond(
                            tf.greater(num_reg_boxes, 0),
                            true_fn=lambda: elem_reg_loss / num_reg_boxes,
                            false_fn=lambda: elem_reg_loss * 0.0,
                        )
                        tf.summary.scalar("{}_regression".format(idx), elem_reg_loss)
                    """
                with tf.variable_scope("reg_norm"):
                    # normalize by the number of positive boxes
                    # with tf.control_dependencies(
                    #    [tf.assert_positive(num_reg_boxes)]):
                    #    regression_loss /= num_reg_boxes
                    regression_loss = tf.cond(
                        tf.greater(num_reg_boxes, 0),
                        true_fn=lambda: regression_loss / num_reg_boxes,
                        false_fn=lambda: regression_loss * 0.0,
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

    def create_path_drop_masks(self, p_img, p_bev, random_values):
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

        def keep_branch():
            return tf.constant(1.0)

        def kill_branch():
            return tf.constant(0.0)

        # The logic works as follows:
        # We have flipped 3 coins, first determines the chance of keeping
        # the image branch, second determines keeping bev branch, the third
        # makes the final decision in the case where both branches were killed
        # off, otherwise the initial img and bev chances are kept.

        img_chances = tf.case(
            [(tf.less(random_values[0], p_img), keep_branch)], default=kill_branch
        )

        bev_chances = tf.case(
            [(tf.less(random_values[1], p_bev), keep_branch)], default=kill_branch
        )

        # Decision to determine whether both branches were killed off
        third_flip = tf.logical_or(
            tf.cast(img_chances, dtype=tf.bool), tf.cast(bev_chances, dtype=tf.bool)
        )
        third_flip = tf.cast(third_flip, dtype=tf.float32)

        # Make a second choice, for the third case
        # Here we use a 50/50 chance to keep either image or bev
        # If its greater than 0.5, keep the image
        img_second_flip = tf.case(
            [(tf.greater(random_values[2], 0.5), keep_branch)], default=kill_branch
        )
        # If its less than or equal to 0.5, keep bev
        bev_second_flip = tf.case(
            [(tf.less_equal(random_values[2], 0.5), keep_branch)], default=kill_branch
        )

        # Use lambda since this returns another condition and it needs to
        # be callable
        final_img_mask = tf.case(
            [(tf.equal(third_flip, 1), lambda: img_chances)],
            default=lambda: img_second_flip,
        )

        final_bev_mask = tf.case(
            [(tf.equal(third_flip, 1), lambda: bev_chances)],
            default=lambda: bev_second_flip,
        )

        return final_img_mask, final_bev_mask
