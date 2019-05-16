import numpy as np
import tensorflow as tf

from avod.builders import feature_extractor_builder
from avod.core import bin_based_box3d_encoder
from avod.core import constants
from avod.core import losses
from avod.core import model
from avod.core import pointfly as pf
from avod.core.anchor_generators import grid_anchor_3d_generator
from avod.core import compute_iou
from avod.core.models import model_util


class RpnModel(model.DetectionModel):
    ##############################
    # Keys for Placeholders
    ##############################
    PL_PC_INPUTS = "pc_inputs_pl"
    PL_LABEL_SEGS = "label_segs_pl"
    PL_LABEL_REGS = "label_regs_pl"
    PL_LABEL_BOXES = "label_boxes_pl"

    ##############################
    # Keys for Predictions
    ##############################
    PRED_SEG_SOFTMAX = "rpn_seg_softmax"
    PRED_SEG_GT = "rpn_seg_gt"

    PRED_CLS = "rpn_cls"
    PRED_REG = "rpn_reg"
    PRED_CLS_GT = "rpn_cls_gt"
    PRED_REG_GT = "rpn_reg_gt"

    PRED_PROPOSALS = "rpn_proposals"
    PRED_OBJECTNESS_SOFTMAX = "rpn_objectness_softmax"
    PRED_NUM_PROPOSALS_BEFORE_PADDING = "rpn_num_proposals_before_padding"

    PRED_IOU_2D = "rpn_proposal_gt_iou_2d"
    PRED_IOU_3D = "rpn_proposal_gt_iou_3d"

    ##############################
    # Keys for SAVE RPN FEATURE
    ##############################
    SAVE_RPN_PTS = "save_rpn_pts"
    SAVE_RPN_FTS = "save_rpn_fts"
    SAVE_RPN_INTENSITY = "save_rpn_intensity"
    SAVE_RPN_FG_MASK = "save_rpn_fg_mask"

    ##############################
    # Keys for Loss
    ##############################
    LOSS_RPN_SEGMENTATION = "rpn_seg_loss"
    LOSS_RPN_BIN_CLASSIFICATION = "rpn_cls_loss"
    LOSS_RPN_REGRESSION = "rpn_reg_loss"

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
            raise ValueError(
                "Invalid train_val_test value,"
                'should be one of ["train", "val", "test"]'
            )
        self._train_val_test = train_val_test

        self._is_training = self._train_val_test == "train"

        # Input config
        input_config = self._config.input_config
        self._pc_sample_pts = input_config.pc_sample_pts
        self._pc_data_dim = input_config.pc_data_dim
        self._pc_sample_pts_variance = input_config.pc_sample_pts_variance
        self._pc_sample_pts_clip = input_config.pc_sample_pts_clip
        self.NUM_FG_POINT = 2048
        # self._img_pixel_size = np.asarray([input_config.img_dims_h,
        #                                   input_config.img_dims_w])
        # self._img_depth = input_config.img_depth

        # Rpn config
        rpn_config = self._config.rpn_config
        self._use_intensity_feature = rpn_config.rpn_use_intensity_feature
        self._fusion_method = rpn_config.rpn_fusion_method
        self._fixed_num_proposal_nms = rpn_config.rpn_fixed_num_proposal_nms

        if self._train_val_test in ["train", "val"]:
            self._pre_nms_size = rpn_config.rpn_train_pre_nms_size
            self._post_nms_size = rpn_config.rpn_train_post_nms_size
            self._nms_iou_thresh = rpn_config.rpn_train_nms_iou_thresh
        else:
            self._pre_nms_size = rpn_config.rpn_test_pre_nms_size
            self._post_nms_size = rpn_config.rpn_test_post_nms_size
            self._nms_iou_thresh = rpn_config.rpn_test_nms_iou_thresh

        assert (
            self._pre_nms_size >= self._post_nms_size
        ), "post nms size must be no greater than pre nms size"

        self.S = rpn_config.rpn_xz_search_range
        self.DELTA = rpn_config.rpn_xz_bin_len
        self.NUM_BIN_X = int(2 * self.S / self.DELTA)
        self.NUM_BIN_Z = self.NUM_BIN_X

        self.R = rpn_config.rpn_theta_search_range * np.pi
        self.DELTA_THETA = 2 * self.R / rpn_config.rpn_theta_bin_num
        self.NUM_BIN_THETA = rpn_config.rpn_theta_bin_num

        # Feature Extractor Nets
        self._pc_feature_extractor = feature_extractor_builder.get_extractor(
            self._config.layers_config.pc_feature_extractor
        )
        """
        self._img_feature_extractor = \
            feature_extractor_builder.get_extractor(
                self._config.layers_config.img_feature_extractor)
        """
        # Network input placeholders
        self.placeholders = dict()

        # Inputs to network placeholders
        self._placeholder_inputs = dict()

        # Information about the current sample
        self._sample_names = []

        # Dataset
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        # Overwrite the dataset's variable with the config
        self.dataset.train_val_test = self._train_val_test
        self._area_extents = self.dataset.kitti_utils.area_extents
        self._bev_extents = self.dataset.kitti_utils.bev_extents
        self._cluster_sizes, _ = self.dataset.get_cluster_info()
        self._anchor_strides = self.dataset.kitti_utils.anchor_strides
        self._anchor_generator = grid_anchor_3d_generator.GridAnchor3dGenerator()

        # self._path_drop_probabilities = self._config.path_drop_probabilities
        self._train_on_all_samples = self._config.train_on_all_samples
        self._eval_all_samples = self._config.eval_all_samples
        # Overwrite the dataset's variable with the config
        self.dataset.train_on_all_samples = self._train_on_all_samples
        self.dataset.eval_all_samples = self._eval_all_samples
        """
        if self._train_val_test in ["val", "test"]:
            # Disable path-drop, this should already be disabled inside the
            # evaluator, but just in case.
            self._path_drop_probabilities[0] = 1.0
            self._path_drop_probabilities[1] = 1.0
        """

    def _add_placeholder(self, dtype, shape, name):
        placeholder = tf.placeholder(dtype, shape, name)
        self.placeholders[name] = placeholder
        return placeholder

    def _set_up_input_pls(self):
        """Sets up input placeholders by adding them to self._placeholders.
        Keys are defined as self.PL_*.
        """
        with tf.variable_scope("pc_input"):
            # Placeholder for PC input, to be filled in with feed_dict
            pc_input_placeholder = self._add_placeholder(
                tf.float32,
                (self._batch_size, self._pc_sample_pts, self._pc_data_dim),
                self.PL_PC_INPUTS,
            )  # (B,P,C)

            self._pc_pts_preprocessed, self._pc_intensities = self._pc_feature_extractor.preprocess_input(
                pc_input_placeholder, self._config.input_config, self._is_training
            )

        with tf.variable_scope("pl_labels"):
            self._add_placeholder(
                tf.float32, [self._batch_size, self._pc_sample_pts], self.PL_LABEL_SEGS
            )  # (B,P)

            self._add_placeholder(
                tf.float32,
                [self._batch_size, self._pc_sample_pts, 7],
                self.PL_LABEL_REGS,
            )  # (B,P,7)

            self._add_placeholder(
                tf.float32, [self._batch_size, None, 7], self.PL_LABEL_BOXES
            )  # (B,m,7)

    def _set_up_feature_extractors(self):
        """Sets up feature extractors and stores feature maps and
        bottlenecks as member variables.
        """
        self._pc_pts, self._pc_fts = self._pc_feature_extractor.build(
            self._pc_pts_preprocessed,
            self._pc_intensities if self._use_intensity_feature else None,
            self._is_training,
        )  # (B,P,3) (B,P,C)

        tf.summary.histogram("pc_fts", self._pc_fts)

    def _gather_residuals(
        self, res_x_norms, res_z_norms, res_theta_norms, bin_x, bin_z, bin_theta
    ):

        """
        Input:
            res_x_norms: (B,p,K)
            bin_x:(B,p)
        return:
            res_x_norm: (B,p)

        TF version: (if p is not None)
        ##########
        """
        B = bin_x.shape[0].value
        p = bin_x.shape[1].value  # maybe None
        Bs = tf.range(B)
        ps = tf.range(p)
        mB, mp = tf.meshgrid(Bs, ps)
        Bp = tf.stack([tf.transpose(mB), tf.transpose(mp)], axis=2)  # (B,p,2)

        BpK_x = tf.concat([Bp, tf.reshape(bin_x, [B, p, 1])], axis=2)  # (B,p,3)
        res_x_norm = tf.gather_nd(res_x_norms, BpK_x)  # (B,p)

        BpK_z = tf.concat([Bp, tf.reshape(bin_z, [B, p, 1])], axis=2)  # (B,p,3)
        res_z_norm = tf.gather_nd(res_z_norms, BpK_z)  # (B,p)

        BpK_theta = tf.concat([Bp, tf.reshape(bin_theta, [B, p, 1])], axis=2)  # (B,p,3)
        res_theta_norm = tf.gather_nd(res_theta_norms, BpK_theta)  # (B,p)

        """
        NumPy version: if p is None, by using tf.py_func, p should be determined
        #############
        res_x_norm = np.take_along_axis(res_x_norms, np.expand_dims(bin_x, -1), axis=-1) #(B,p,1)
        res_x_norm = np.squeeze(res_x_norm, -1)
        res_z_norm = np.take_along_axis(res_z_norms, np.expand_dims(bin_z, -1), axis=-1) #(B,p,1)
        res_z_norm = np.squeeze(res_z_norm, -1)
        res_theta_norm = np.take_along_axis(res_theta_norms, np.expand_dims(bin_theta, -1), axis=-1) #(B,p,1)
        res_theta_norm = np.squeeze(res_theta_norm, -1)
        """

        return res_x_norm, res_z_norm, res_theta_norm

    def _gather_mean_sizes(self, cluster_sizes, cls_preds):
        """
        Input:
            cluster_sizes: (Klass, Cluster=1, 3) [l,w,h], Klass is 0-based
            cls_preds: (B,p), [klass], kclass is 1-based, 0-background
        Output
            mean_sizes: (B,p,3) [l,w,h]

        TF version: (if p is not None)
        ##########
        """
        B = cls_preds.shape[0].value
        p = cls_preds.shape[1].value

        Bs = tf.range(B)
        ps = tf.range(p)
        mB, mp = tf.meshgrid(Bs, ps)
        Bp = tf.stack([tf.transpose(mB), tf.transpose(mp)], axis=2)  # (B,p,2)

        K_mean_sizes = tf.reshape(cluster_sizes, [-1, 3])
        pK_mean_sizes = tf.tile(tf.expand_dims(K_mean_sizes, 0), [p, 1, 1])
        BpK_mean_sizes = tf.tile(tf.expand_dims(pK_mean_sizes, 0), [B, 1, 1, 1])

        BpK = tf.concat([Bp, tf.reshape(cls_preds, [B, p, 1])], axis=2)  # (B,p,3)

        mean_sizes = tf.gather_nd(BpK_mean_sizes, BpK)

        """
        NumPy version: if p is None, by using tf.py_func, p should be determined
        #############
        K_mean_sizes = np.reshape(cluster_sizes, (-1,3))
        K_mean_sizes = np.vstack([np.asarray([0.0, 0.0, 0.0]), K_mean_sizes]) # insert 0-background
        mean_sizes = K_mean_sizes[cls_preds]
        return mean_sizes.astype(np.float32)
        """

        return mean_sizes

    def build(self, **kwargs):

        # Setup input placeholders
        self._set_up_input_pls()

        # Setup feature extractors
        self._set_up_feature_extractors()

        # branch-1: foreground point segmentation
        #########################################
        with tf.variable_scope("foreground_segmentation"):
            seg_logits = pf.dense(
                self._pc_fts,
                2,
                "seg_logits",
                self._is_training,
                with_bn=False,
                activation=None,
            )  # (B,P,K)
            seg_softmax = tf.nn.softmax(seg_logits, name="seg_softmax")  # (B,P,K)
            seg_preds = tf.argmax(
                seg_softmax, axis=-1, name="seg_predictions", output_type=tf.int32
            )  # (B,P)
            seg_scores = tf.reduce_max(
                seg_softmax[:, :, 1:], axis=-1, name="seg_scores"
            )  # (B,P)

        label_cls = self.placeholders[self.PL_LABEL_SEGS]  # (B,P)
        label_reg = self.placeholders[self.PL_LABEL_REGS]  # (B,P,7)

        proposal_pts = self._pc_pts
        proposal_fts = self._pc_fts
        proposal_preds = seg_preds
        proposal_scores = seg_scores
        proposal_label_reg = label_reg

        # foreground point masking
        with tf.variable_scope("foreground_masking"):
            if self._train_val_test in ["train", "val"]:
                self._foreground_mask = label_cls > 0  # (B,P)
            else:
                self._foreground_mask = seg_preds > 0  # (B,P)

            if (
                self._train_val_test in ["val", "test"]
                and not self._fixed_num_proposal_nms
            ):
                proposal_pts, proposal_fts, proposal_preds, proposal_scores, proposal_label_reg = model_util.foreground_masking(
                    self._foreground_mask,
                    self.NUM_FG_POINT,
                    self._batch_size,
                    self._pc_pts,
                    self._pc_fts,
                    seg_preds,
                    seg_scores,
                    label_reg,
                )

        # branch-2: bin-based 3D proposal generation
        #########################################
        with tf.variable_scope("bin_based_rpn"):
            # Parse rpn layers config
            fc_layers = [proposal_fts]
            layers_config = self._config.layers_config.rpn_config.fc_layer
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

        bin_x_logits, res_x_norms, bin_z_logits, res_z_norms, bin_theta_logits, res_theta_norms, res_y, res_size = self._parse_rpn_output(
            fc_output
        )
        res_y = tf.squeeze(res_y, [-1])

        # Return the proposals
        ######################################################
        with tf.variable_scope("proposals"):
            bin_x = tf.argmax(bin_x_logits, axis=-1, output_type=tf.int32)  # (B,P)
            bin_z = tf.argmax(bin_z_logits, axis=-1, output_type=tf.int32)  # (B,P)
            bin_theta = tf.argmax(
                bin_theta_logits, axis=-1, output_type=tf.int32
            )  # (B,P)

            res_x_norm, res_z_norm, res_theta_norm = self._gather_residuals(
                res_x_norms, res_z_norms, res_theta_norms, bin_x, bin_z, bin_theta
            )

            mean_sizes = self._gather_mean_sizes(
                tf.convert_to_tensor(np.asarray(self._cluster_sizes, dtype=np.float32)),
                seg_preds,
            )

            # NMS
            if self._train_val_test == "train":
                # to speed up training, skip NMS, as we don't care what top_* is during training
                print("Skip RPN-NMS during training")
            else:
                # Decode bin-based 3D Box
                with tf.variable_scope("decoding"):
                    proposals = bin_based_box3d_encoder.tf_decode(
                        proposal_pts,
                        0,
                        bin_x,
                        res_x_norm,
                        bin_z,
                        res_z_norm,
                        bin_theta,
                        res_theta_norm,
                        res_y,
                        res_size,
                        mean_sizes,
                        self.S,
                        self.DELTA,
                        self.R,
                        self.DELTA_THETA,
                    )  # (B,P,7)

                if self._fixed_num_proposal_nms:
                    confidences = proposal_scores
                    # get _pre_nms_size number of proposals for NMS
                    _, sorted_idxs = tf.nn.top_k(
                        confidences, k=self._pre_nms_size, sorted=True
                    )

                    pre_nms_proposals, pre_nms_confidences = tf.map_fn(
                        model_util.gather_top_n,
                        elems=(proposals, confidences, sorted_idxs),
                        dtype=(tf.float32, tf.float32, tf.float32),
                    )
                else:
                    bin_x_scores = tf.reduce_max(tf.nn.softmax(bin_x_logits), axis=-1)
                    bin_z_scores = tf.reduce_max(tf.nn.softmax(bin_z_logits), axis=-1)
                    bin_theta_scores = tf.reduce_max(
                        tf.nn.softmax(bin_theta_logits), axis=-1
                    )

                    confidences = (
                        proposal_scores * bin_x_scores * bin_z_scores * bin_theta_scores
                    )  # (B,P)
                    pre_nms_proposals, pre_nms_confidences = (proposals, confidences)

                # oriented-NMS is much slower than non-oriented-NMS (tf.image.non_max_suppression)
                # while get significant higher proposal recall@IoU=0.7
                oriented_NMS = True
                print("RPN oriented_NMS = " + str(oriented_NMS))
                # BEV-NMS and ignore multiclass
                with tf.variable_scope("bev_nms"):

                    def sb_nms_fn(x):
                        return model_util.sb_nms_fn(
                            x,
                            oriented_NMS,
                            self._nms_iou_thresh,
                            self._post_nms_size,
                            self._fixed_num_proposal_nms,
                            self._bev_extents,
                        )

                    nms_indices, num_proposals_before_padding = tf.map_fn(
                        sb_nms_fn,
                        elems=[pre_nms_proposals, pre_nms_confidences],
                        dtype=(tf.int32, tf.int32),
                    )
                    post_nms_proposals, post_nms_confidences = tf.map_fn(
                        model_util.sb_nms_selection,
                        elems=(pre_nms_proposals, pre_nms_confidences, nms_indices),
                        dtype=(tf.float32, tf.float32),
                    )

                # Compute IOUs
                if self._train_val_test == "val":
                    with tf.variable_scope("compute_ious"):
                        iou3ds, iou2ds = tf.map_fn(
                            model_util.sb_compute_iou,
                            elems=(
                                post_nms_proposals,
                                self.placeholders[self.PL_LABEL_BOXES],
                            ),
                            dtype=(tf.float32, tf.float32),
                        )

        if self._train_val_test in ["train", "val"]:
            ######################################################
            # GTs for the loss function & metrics
            ######################################################

            # Ground Truth Seg
            with tf.variable_scope("seg_one_hot_classes"):
                segs_gt_one_hot = tf.one_hot(
                    tf.to_int32(label_cls), 2, on_value=1.0, off_value=0.0
                )

            with tf.variable_scope("segmentation_accuracy"):
                avg_num_foreground_pts = (
                    tf.reduce_sum(tf.cast(self._foreground_mask, tf.float32))
                    / self._batch_size
                )
                tf.summary.scalar("avg_foreground_points_num", avg_num_foreground_pts)
                # seg accuracy
                seg_correct = tf.equal(seg_preds, tf.to_int32(label_cls))
                seg_accuracy = tf.reduce_mean(tf.to_float(seg_correct))
                tf.summary.scalar("segmentation_accuracy", seg_accuracy)

            # Ground Truth Box Cls/Reg
            with tf.variable_scope("box_cls_reg_gt"):
                (
                    bin_x_gt,
                    res_x_gt,
                    bin_z_gt,
                    res_z_gt,
                    bin_theta_gt,
                    res_theta_gt,
                    res_y_gt,
                    res_size_gt,
                ) = bin_based_box3d_encoder.tf_encode(
                    proposal_pts,
                    0,
                    proposal_label_reg,
                    mean_sizes,
                    self.S,
                    self.DELTA,
                    self.R,
                    self.DELTA_THETA,
                )

                bin_x_gt_one_hot, bin_z_gt_one_hot, bin_theta_gt_one_hot = model_util.x_z_theta_one_hot_encoding(
                    bin_x_gt,
                    bin_z_gt,
                    bin_theta_gt,
                    self.S,
                    self.DELTA,
                    self.R,
                    self.DELTA_THETA,
                )

            ######################################################
            # Prediction Dict
            ######################################################
            predictions = dict()
            predictions[self.PRED_SEG_SOFTMAX] = seg_softmax
            predictions[self.PRED_SEG_GT] = segs_gt_one_hot

            # Foreground BOX predictions
            predictions[self.PRED_CLS] = (bin_x_logits, bin_z_logits, bin_theta_logits)
            predictions[self.PRED_REG] = (
                res_x_norm,
                res_z_norm,
                res_theta_norm,
                res_y,
                res_size,
            )

            # Foreground BOX ground truth
            predictions[self.PRED_CLS_GT] = (
                bin_x_gt_one_hot,
                bin_z_gt_one_hot,
                bin_theta_gt_one_hot,
            )
            predictions[self.PRED_REG_GT] = (
                res_x_gt,
                res_z_gt,
                res_theta_gt,
                res_y_gt,
                res_size_gt,
            )
            if self._train_val_test == "val":
                predictions[self.PRED_IOU_2D] = iou2ds
                predictions[self.PRED_IOU_3D] = iou3ds
                predictions[self.PRED_PROPOSALS] = post_nms_proposals
                predictions[self.PRED_OBJECTNESS_SOFTMAX] = post_nms_confidences
                predictions[
                    self.PRED_NUM_PROPOSALS_BEFORE_PADDING
                ] = num_proposals_before_padding
        else:
            # self._train_val_test == 'test'
            predictions[self.PRED_SEG_SOFTMAX] = seg_softmax
            predictions[self.PRED_PROPOSALS] = post_nms_proposals
            predictions[self.PRED_OBJECTNESS_SOFTMAX] = post_nms_confidences
            predictions[
                self.PRED_NUM_PROPOSALS_BEFORE_PADDING
            ] = num_proposals_before_padding

        if (
            self._train_val_test in ["val", "test"]
            and "save_rpn_feature" in kwargs
            and kwargs["save_rpn_feature"]
        ):
            predictions[self.SAVE_RPN_PTS] = self._pc_pts
            predictions[self.SAVE_RPN_FTS] = self._pc_fts
            predictions[self.SAVE_RPN_INTENSITY] = self._pc_intensities
            predictions[self.SAVE_RPN_FG_MASK] = self._foreground_mask
        return predictions

    def _parse_rpn_output(self, rpn_output):
        """
        Input:
            rpn_output: (B, p, NUM_BIN_X*2 + NUM_BIN_Z*2 + NUM_BIN_THETA*2 + 4)
        Output:
            bin_x_logits
            res_x_norms

            bin_z_logits
            res_z_norms

            bin_theta_logits
            res_theta_norms

            res_y

            res_size: (l,w,h)
        """
        bin_x_logits = tf.slice(rpn_output, [0, 0, 0], [-1, -1, self.NUM_BIN_X])
        res_x_norms = tf.slice(
            rpn_output, [0, 0, self.NUM_BIN_X], [-1, -1, self.NUM_BIN_X]
        )

        bin_z_logits = tf.slice(
            rpn_output, [0, 0, self.NUM_BIN_X * 2], [-1, -1, self.NUM_BIN_Z]
        )
        res_z_norms = tf.slice(
            rpn_output,
            [0, 0, self.NUM_BIN_X * 2 + self.NUM_BIN_Z],
            [-1, -1, self.NUM_BIN_Z],
        )

        bin_theta_logits = tf.slice(
            rpn_output,
            [0, 0, self.NUM_BIN_X * 2 + self.NUM_BIN_Z * 2],
            [-1, -1, self.NUM_BIN_THETA],
        )
        res_theta_norms = tf.slice(
            rpn_output,
            [0, 0, self.NUM_BIN_X * 2 + self.NUM_BIN_Z * 2 + self.NUM_BIN_THETA],
            [-1, -1, self.NUM_BIN_THETA],
        )

        res_y = tf.slice(
            rpn_output,
            [0, 0, self.NUM_BIN_X * 2 + self.NUM_BIN_Z * 2 + self.NUM_BIN_THETA * 2],
            [-1, -1, 1],
        )

        res_size = tf.slice(
            rpn_output,
            [
                0,
                0,
                self.NUM_BIN_X * 2 + self.NUM_BIN_Z * 2 + self.NUM_BIN_THETA * 2 + 1,
            ],
            [-1, -1, 3],
        )

        return (
            bin_x_logits,
            res_x_norms,
            bin_z_logits,
            res_z_norms,
            bin_theta_logits,
            res_theta_norms,
            res_y,
            res_size,
        )

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
                    batch_size, True, model="rpn", pc_sample_pts=self._pc_sample_pts
                )

            else:  # self._train_val_test == "val"
                # Load samples in order for validation
                batch_data, sample_names = self.dataset.next_batch(
                    batch_size, False, model="rpn", pc_sample_pts=self._pc_sample_pts
                )
        else:
            # For testing, any sample should work
            if sample_index is not None:
                samples = self.dataset.load_samples(
                    [sample_index], model="rpn", pc_sample_pts=self._pc_sample_pts
                )
                batch_data, sample_names = self.dataset.collate_batch(samples)
            else:
                batch_data, sample_names = self.dataset.next_batch(
                    batch_size, False, model="rpn", pc_sample_pts=self._pc_sample_pts
                )

        # this is a list to match the explicit shape for the placeholder
        # self._placeholder_inputs[self.PL_IMG_IDX] = [int(sample_name)]

        # Fill in the rest
        # self._placeholder_inputs[self.PL_IMG_INPUT] = image_input
        self._placeholder_inputs[self.PL_PC_INPUTS] = batch_data[
            constants.KEY_POINT_CLOUD
        ]
        self._placeholder_inputs[self.PL_LABEL_SEGS] = batch_data[
            constants.KEY_LABEL_SEG
        ]
        self._placeholder_inputs[self.PL_LABEL_REGS] = batch_data[
            constants.KEY_LABEL_REG
        ]
        self._placeholder_inputs[self.PL_LABEL_BOXES] = batch_data[
            constants.KEY_LABEL_BOXES_3D
        ]
        # Sample Info
        self._sample_names = sample_names
        # img_idx is a list to match the placeholder shape
        # self._placeholder_inputs[self.PL_IMG_IDX] = [int(sample_name)]

        # Create a feed_dict and fill it with input values
        feed_dict = dict()
        for key, value in self.placeholders.items():
            feed_dict[value] = self._placeholder_inputs[key]

        return feed_dict

    def loss(self, prediction_dict):

        seg_softmax = prediction_dict[self.PRED_SEG_SOFTMAX]
        seg_gt = prediction_dict[self.PRED_SEG_GT]

        with tf.variable_scope("rpn_losses"):
            with tf.variable_scope("segmentation"):
                seg_loss = losses.WeightedFocalLoss()
                seg_loss_weight = self._config.loss_config.seg_loss_weight
                segmentation_loss = seg_loss(
                    seg_softmax, seg_gt, weight=seg_loss_weight
                )
                with tf.variable_scope("seg_norm"):
                    num_total_pts = self._batch_size * self._pc_sample_pts
                    segmentation_loss /= num_total_pts
                    tf.summary.scalar("segmentation", segmentation_loss)

            # these should include foreground pts only
            with tf.variable_scope("bin_classification"):
                cls_loss = losses.WeightedSoftmaxLoss()
                cls_loss_weight = self._config.loss_config.cls_loss_weight
                bin_classification_loss = 0.0
                # bin_x_logits, bin_z_logits, bin_theta_logits = prediction_dict[self.PRED_FG_CLS]
                # bin_x_gt_one_hot, bin_z_gt_one_hot, bin_theta_gt_one_hot = prediction_dict[self.PRED_FG_CLS_GT]
                for elem in zip(
                    prediction_dict[self.PRED_CLS], prediction_dict[self.PRED_CLS_GT]
                ):
                    foreground_pred_cls = tf.boolean_mask(
                        elem[0], self._foreground_mask
                    )
                    foreground_pred_cls_gt = tf.boolean_mask(
                        elem[1], self._foreground_mask
                    )
                    bin_classification_loss += cls_loss(
                        foreground_pred_cls,
                        foreground_pred_cls_gt,
                        weight=cls_loss_weight,
                    )
                with tf.variable_scope("cls_norm"):
                    # normalize by the number of foreground pts
                    num_foreground_pts = tf.reduce_sum(
                        tf.cast(self._foreground_mask, tf.float32)
                    )
                    bin_classification_loss = tf.cond(
                        tf.greater(num_foreground_pts, 0),
                        true_fn=lambda: bin_classification_loss / num_foreground_pts,
                        false_fn=lambda: bin_classification_loss * 0,
                    )
                    tf.summary.scalar("bin_classification", bin_classification_loss)

            with tf.variable_scope("regression"):
                reg_loss = losses.WeightedSmoothL1Loss()
                reg_loss_weight = self._config.loss_config.reg_loss_weight
                regression_loss = 0.0
                # res_x_norm, res_z_norm, res_theta_norm, res_y, res_size = prediction_dict[self.PRED_FG_REG]
                # res_x_gt, res_z_gt, res_theta_gt, res_y_gt, res_size_gt = prediction_dict[self.PRED_FG_REG_GT]
                for elem in zip(
                    prediction_dict[self.PRED_REG], prediction_dict[self.PRED_REG_GT]
                ):
                    foreground_pred_reg = tf.boolean_mask(
                        elem[0], self._foreground_mask
                    )
                    foreground_pred_reg_gt = tf.boolean_mask(
                        elem[1], self._foreground_mask
                    )
                    regression_loss += reg_loss(
                        foreground_pred_reg,
                        foreground_pred_reg_gt,
                        weight=reg_loss_weight,
                    )
                with tf.variable_scope("reg_norm"):
                    # normalize by the number of foreground pts
                    regression_loss = tf.cond(
                        tf.greater(num_foreground_pts, 0),
                        true_fn=lambda: regression_loss / num_foreground_pts,
                        false_fn=lambda: regression_loss * 0,
                    )
                    tf.summary.scalar("regression", regression_loss)

            with tf.variable_scope("rpn_loss"):
                rpn_loss = segmentation_loss + bin_classification_loss + regression_loss

        loss_dict = {
            self.LOSS_RPN_SEGMENTATION: segmentation_loss,
            self.LOSS_RPN_BIN_CLASSIFICATION: bin_classification_loss,
            self.LOSS_RPN_REGRESSION: regression_loss,
        }

        return loss_dict, rpn_loss
