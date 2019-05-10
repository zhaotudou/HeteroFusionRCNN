"""Dataset utils for preparing data for the network."""

import itertools
import fnmatch
import os

import numpy as np
import cv2

from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection import obj_utils

from avod.core import box_3d_encoder
from avod.core import box_8c_encoder
from avod.core import box_util
from avod.core import constants
from avod.datasets.kitti import kitti_aug
from avod.datasets.kitti.kitti_utils import KittiUtils


class Sample:
    def __init__(self, name, augs):
        self.name = name
        self.augs = augs


class KittiDataset:
    def __init__(self, dataset_config):
        """
        Initializes directories, and loads the sample list

        Args:
            dataset_config: KittiDatasetConfig
                name: unique name for the dataset
                data_split: "train", "val", "test", "trainval"
                data_split_dir: must be specified for custom "training"
                dataset_dir: Kitti dataset dir if not in default location
                classes: relevant classes
                num_clusters: number of k-means clusters to separate for
                    each class
        """
        # Parse config
        self.config = dataset_config

        self.name = self.config.name
        self.data_split = self.config.data_split
        self.dataset_dir = os.path.expanduser(self.config.dataset_dir)
        data_split_dir = self.config.data_split_dir

        self.has_labels = self.config.has_labels
        self.cluster_split = self.config.cluster_split

        self.classes = list(self.config.classes)
        self.num_classes = len(self.classes)
        self.num_clusters = np.asarray(self.config.num_clusters)

        self.aug_list = self.config.aug_list
        self.reg_aug_method = self.config.reg_aug_method

        # Determines the network mode. This is initialized to 'train' but
        # is overwritten inside the model based on the mode.
        self.train_val_test = "train"
        # Determines if training includes all samples, including the ones
        # without anchor_info. This is initialized to False, but is overwritten
        # via the config inside the model.
        self.train_on_all_samples = False
        self.eval_all_samples = False

        self._set_up_classes_name()

        # Check that paths and split are valid
        self._check_dataset_dir()

        # Get all files and folders in dataset directory
        all_files = os.listdir(self.dataset_dir)

        # Get possible data splits from txt files in dataset folder
        possible_splits = []
        for file_name in all_files:
            if fnmatch.fnmatch(file_name, "*.txt"):
                possible_splits.append(os.path.splitext(file_name)[0])
        # This directory contains a readme.txt file, remove it from the list
        if "readme" in possible_splits:
            possible_splits.remove("readme")

        if self.data_split not in possible_splits:
            raise ValueError(
                "Invalid data split: {}, possible_splits: {}".format(
                    self.data_split, possible_splits
                )
            )

        # Check data_split_dir
        # Get possible data split dirs from folder names in dataset folder
        possible_split_dirs = []
        for folder_name in all_files:
            if os.path.isdir(self.dataset_dir + "/" + folder_name):
                possible_split_dirs.append(folder_name)
        if data_split_dir in possible_split_dirs:
            split_dir = self.dataset_dir + "/" + data_split_dir
            self._data_split_dir = split_dir
        else:
            raise ValueError(
                "Invalid data split dir: {}, possible dirs: {}".format(
                    data_split_dir, possible_split_dirs
                )
            )

        # Batch pointers
        self._index_in_epoch = 0
        self.epochs_completed = 0

        self._cam_idx = 2

        # Initialize the sample list
        loaded_sample_names = self.load_sample_names(self.data_split)

        # Augment the sample list
        aug_sample_list = []

        # Loop through augmentation lengths e.g.
        # 0: []
        # 1: ['flip'], ['pca_jitter']
        # 2: ['flip', 'pca_jitter']
        for aug_idx in range(len(self.aug_list) + 1):
            # Get all combinations
            augmentations = list(itertools.combinations(self.aug_list, aug_idx))
            for augmentation in augmentations:
                for sample_name in loaded_sample_names:
                    aug_sample_list.append(Sample(sample_name, augmentation))

        self.sample_list = np.asarray(aug_sample_list)
        self.num_samples = len(self.sample_list)
        print("Number of samples in dataset: ", self.num_samples)

        self._set_up_directories()

        # Setup utils object
        self.kitti_utils = KittiUtils(self)

        # mini batch config
        mini_batch_config = dataset_config.mini_batch_config
        cls_iou_thresholds = mini_batch_config.cls_iou_3d_thresholds
        reg_iou_thresholds = mini_batch_config.reg_iou_3d_thresholds
        self.cls_neg_iou_range = [
            cls_iou_thresholds.neg_iou_lo,
            cls_iou_thresholds.neg_iou_hi,
        ]
        self.cls_pos_iou_range = [
            cls_iou_thresholds.pos_iou_lo,
            cls_iou_thresholds.pos_iou_hi,
        ]
        self.reg_neg_iou_range = [
            reg_iou_thresholds.neg_iou_lo,
            reg_iou_thresholds.neg_iou_hi,
        ]
        self.reg_pos_iou_range = [
            reg_iou_thresholds.pos_iou_lo,
            reg_iou_thresholds.pos_iou_hi,
        ]
        self.roi_per_sample = mini_batch_config.roi_per_sample
        self.fg_ratio = mini_batch_config.fg_ratio
        self.hard_bg_ratio = mini_batch_config.hard_bg_ratio

    # Paths
    @property
    def rgb_image_dir(self):
        return self.image_dir

    @property
    def sample_names(self):
        # This is a property since the sample list gets shuffled for training
        return np.asarray([sample.name for sample in self.sample_list])

    def _check_dataset_dir(self):
        """Checks that dataset directory exists in the file system

        Raises:
            FileNotFoundError: if the dataset folder is missing
        """
        # Check that dataset path is valid
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(
                "Dataset path does not exist: {}".format(self.dataset_dir)
            )

    def _set_up_directories(self):
        """Sets up data directories."""
        # Setup Directories
        self.image_dir = self._data_split_dir + "/image_" + str(self._cam_idx)
        self.calib_dir = self._data_split_dir + "/calib"
        self.disp_dir = self._data_split_dir + "/disparity"
        self.planes_dir = self._data_split_dir + "/planes"
        self.velo_dir = self._data_split_dir + "/velodyne"
        self.depth_dir = self._data_split_dir + "/depth_" + str(self._cam_idx)
        self.proposal_dir = self._data_split_dir + "/proposal"
        self.proposal_iou_dir = self._data_split_dir + "/proposal_iou"
        self.rpn_feature_dir = self._data_split_dir + "/rpn_feature"

        # Labels are always in the training folder
        self.label_dir = self.dataset_dir + "/training/label_" + str(self._cam_idx)

    def _set_up_classes_name(self):
        # Unique identifier for multiple classes
        if self.num_classes > 1:
            if self.classes == ["Pedestrian", "Cyclist"]:
                self.classes_name = "People"
            elif self.classes == ["Car", "Pedestrian", "Cyclist"]:
                self.classes_name = "All"
            else:
                raise NotImplementedError(
                    "Need new unique identifier for " "multiple classes"
                )
        else:
            self.classes_name = self.classes[0]

    # Get sample paths
    def get_rgb_image_path(self, sample_name):
        return self.rgb_image_dir + "/" + sample_name + ".png"

    def get_depth_map_path(self, sample_name):
        return self.depth_dir + "/" + sample_name + "_left_depth.png"

    def get_velodyne_path(self, sample_name):
        return self.velo_dir + "/" + sample_name + ".bin"

    def get_proposal_path(self, sample_name):
        return self.proposal_dir + "/" + sample_name + ".txt"

    def get_rpn_feature_path(self, sample_name):
        return self.rpn_feature_dir + "/" + sample_name + ".npy"

    def get_proposal_iou_path(self, sample_name):
        return self.proposal_iou_dir + "/" + sample_name + ".txt"

    def get_proposal(self, sample_name):
        proposals = np.loadtxt(self.get_proposal_path(sample_name)).reshape((-1, 8))[
            :, 0:7
        ]
        return proposals

    def get_rpn_features(self, sample_name):
        rpn_features = np.load(self.get_rpn_feature_path(sample_name))
        return (
            rpn_features[:, 0:3],
            rpn_features[:, 3],
            rpn_features[:, 4],
            rpn_features[:, 5:],
        )

    def get_proposal_iou(self, sample_name):
        proposals_iou = np.loadtxt(self.get_proposal_iou_path(sample_name))
        return proposals_iou

    # Cluster info
    def get_cluster_info(self):
        return self.kitti_utils.clusters, self.kitti_utils.std_devs

    def get_label_seg(self, sample_name):
        return self.kitti_utils.get_label_seg(
            self.classes_name, self.kitti_utils.expand_gt_size, sample_name
        )

    # Data loading methods
    def load_sample_names(self, data_split):
        """Load the sample names listed in this dataset's set file
        (e.g. train.txt, validation.txt)

        Args:
            data_split: override the sample list to load (e.g. for clustering)

        Returns:
            A list of sample names (file names) read from
            the .txt file corresponding to the data split
        """
        set_file = self.dataset_dir + "/" + data_split + ".txt"
        with open(set_file, "r") as f:
            sample_names = f.read().splitlines()

        return np.array(sample_names)

    def load_samples(self, indices, **kwargs):
        assert "model" in kwargs
        model = kwargs.pop("model")
        if model == "rpn":
            return self.load_rpn_samples(indices, **kwargs)
        elif model == "rcnn":
            return self.load_rcnn_samples(indices, **kwargs)
        else:
            raise ValueError("model should be rpn or rcnn")

    def load_rpn_samples(self, indices, pc_sample_pts=16384):
        """ Loads input-output data for a set of samples. Should only be
            called when a particular sample dict is required. Otherwise,
            samples should be provided by the next_batch function

        Args:
            indices: A list of sample indices from the dataset.sample_list
                to be loaded

        Return:
            samples: a list of data sample dicts
        """
        sample_dicts = []
        for sample_idx in indices:
            sample = self.sample_list[sample_idx]

            if self.has_labels:
                obj_labels = obj_utils.read_labels(self.label_dir, int(sample.name))
                # Only use objects that match dataset classes
                obj_labels = self.kitti_utils.filter_labels(obj_labels)
                if len(obj_labels) <= 0:
                    continue

            # Load image (BGR -> RGB)
            cv_bgr_image = cv2.imread(self.get_rgb_image_path(sample.name))
            rgb_image = cv_bgr_image[..., ::-1]
            image_shape = rgb_image.shape[0:2]
            image_input = rgb_image

            # Load PC in rect image space
            pts_rect, pts_intensity = self.kitti_utils.get_point_cloud(
                int(sample.name), image_shape
            )

            if pc_sample_pts < len(pts_rect):
                pts_depth = pts_rect[:, 2]
                pts_near_flag = pts_depth < 40.0
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                near_idxs = np.where(pts_near_flag == 1)[0]
                near_idxs_choice = np.random.choice(
                    near_idxs, pc_sample_pts - len(far_idxs_choice), replace=False
                )

                choice = (
                    np.concatenate((near_idxs_choice, far_idxs_choice), axis=0)
                    if len(far_idxs_choice) > 0
                    else near_idxs_choice
                )
                np.random.shuffle(choice)
            else:
                choice = np.arange(0, len(pts_rect), dtype=np.int32)
                if pc_sample_pts > len(pts_rect):
                    extra_choice = np.random.choice(
                        choice, pc_sample_pts - len(pts_rect), replace=False
                    )
                    choice = np.concatenate((choice, extra_choice), axis=0)
                np.random.shuffle(choice)

            sampled_pts_rect = pts_rect[choice, :]
            sampled_pts_intensity = (
                pts_intensity[choice] - 0.5
            )  # translate intensity to [-0.5, 0.5]
            sampled_pc = np.hstack((sampled_pts_rect, sampled_pts_intensity))

            # Only read labels if they exist
            if self.has_labels:
                # Augmentation (Flipping)
                if kitti_aug.AUG_FLIPPING in sample.augs:
                    # image_input = kitti_aug.flip_image(image_input)
                    sampled_pc = kitti_aug.flip_points(sampled_pc)
                    obj_labels = [
                        kitti_aug.flip_label_in_3d_only(obj) for obj in obj_labels
                    ]

                # Augmentation (Image Jitter)
                if kitti_aug.AUG_PCA_JITTER in sample.augs:
                    image_input[:, :, 0:3] = kitti_aug.apply_pca_jitter(
                        image_input[:, :, 0:3]
                    )

                label_boxes_3d = np.asarray(
                    [
                        box_3d_encoder.object_label_to_box_3d(obj_label)
                        for obj_label in obj_labels
                    ]
                )

                # generate training labels
                label_seg, label_reg = self.generate_rpn_training_labels(
                    sampled_pc[:, :3], label_boxes_3d
                )
            else:
                label_boxes_3d = np.zeros((1, 7))
                label_seg = np.zeros(pc_sample_pts)
                label_reg = np.zeros((pc_sample_pts, 7))

            sample_dict = {
                constants.KEY_LABEL_SEG: label_seg,
                constants.KEY_LABEL_REG: label_reg,
                constants.KEY_LABEL_BOXES_3D: label_boxes_3d,
                constants.KEY_POINT_CLOUD: sampled_pc,
                constants.KEY_SAMPLE_NAME: sample.name,
                constants.KEY_SAMPLE_AUGS: sample.augs,
            }
            sample_dicts.append(sample_dict)

        return sample_dicts

    def generate_rpn_training_labels(self, pts_rect, gt_boxes3d):
        cls_label = np.zeros((pts_rect.shape[0]), dtype=np.int32)
        reg_label = np.zeros((pts_rect.shape[0], 7), dtype=np.float32)
        extend_gt_boxes3d = gt_boxes3d.copy()
        extend_gt_boxes3d[:, 3:6] += self.kitti_utils.expand_gt_size * 2
        extend_gt_boxes3d[:, 1] += self.kitti_utils.expand_gt_size

        gt_corners = box_8c_encoder.np_box_3d_to_box_8co(gt_boxes3d)
        extend_gt_corners = box_8c_encoder.np_box_3d_to_box_8co(extend_gt_boxes3d)

        for k in range(gt_boxes3d.shape[0]):
            box_corners = gt_corners[k]
            fg_pt_flag = obj_utils.is_point_inside(pts_rect.T, box_corners.T)
            cls_label[fg_pt_flag] = 1
            reg_label[fg_pt_flag, :] = gt_boxes3d[k]

            # enlarge the bbox3d, ignore nearby points
            extend_box_corners = extend_gt_corners[k]
            fg_enlarge_flag = obj_utils.is_point_inside(
                pts_rect.T, extend_box_corners.T
            )
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            cls_label[ignore_flag] = -1

        return cls_label, reg_label

    def load_rcnn_samples(self, indices):
        sample_dicts = []
        for sample_idx in indices:
            sample = self.sample_list[sample_idx]

            if self.has_labels:
                obj_labels = obj_utils.read_labels(self.label_dir, int(sample.name))
                # Only use objects that match dataset classes
                obj_labels = self.kitti_utils.filter_labels(obj_labels)
                if len(obj_labels) <= 0:
                    continue

                gt_boxes3d = np.asarray(
                    [
                        box_3d_encoder.object_label_to_box_3d(obj_label)
                        for obj_label in obj_labels
                    ]
                )
                gt_classes = np.asarray(
                    [
                        self.kitti_utils.class_str_to_index(obj_label.type)
                        for obj_label in obj_labels
                    ]
                )

            rpn_pts, rpn_intensity, rpn_fg_mask, rpn_fts = self.get_rpn_features(
                sample.name
            )
            roi_boxes3d = self.get_proposal(sample.name)

            if self.has_labels:
                iou3d = self.get_proposal_iou(sample.name).reshape(
                    (-1, gt_boxes3d.shape[0])
                )
                gt_info = np.hstack((gt_boxes3d, gt_classes.reshape((-1, 1))))
                rois, iou_of_rois, gt_of_rois = self.sample_rois_for_rcnn(
                    roi_boxes3d, iou3d, gt_info
                )
            else:
                rois = np.zeros((self.roi_per_sample, 7))
                iou_of_rois = np.zeros(self.roi_per_sample)
                gt_of_rois = np.zeros((self.roi_per_sample, 8))
                pass

            sample_dict = {
                constants.KEY_RPN_PTS: rpn_pts,
                constants.KEY_RPN_INTENSITY: rpn_intensity,
                constants.KEY_RPN_FG_MASK: rpn_fg_mask,
                constants.KEY_RPN_FTS: rpn_fts,
                constants.KEY_RPN_ROI: rois,
                constants.KEY_RPN_IOU: iou_of_rois,
                constants.KEY_RPN_GT: gt_of_rois,
                constants.KEY_SAMPLE_NAME: sample.name,
                constants.KEY_SAMPLE_AUGS: sample.augs,
            }
            sample_dicts.append(sample_dict)

        return sample_dicts

    def sample_rois_for_rcnn(self, roi_boxes3d, iou3d, gt_info):
        """
        :param roi_boxes3d: (m, 7)
        :param iou3d: (m, n)
        :param gt_info: (n, 8) [x, y, z, h, w, l, ry, cls]
        :return
            rois: (N, 7)
            iou_of_rois: (N)
            gt_of_rois: (N, 8)
        """
        max_overlaps, gt_assignment = iou3d.max(axis=1), iou3d.argmax(axis=1)
        max_iou_of_gt, roi_assignment = iou3d.max(axis=0), iou3d.argmax(axis=0)
        roi_assignment = roi_assignment[max_iou_of_gt > 0].reshape(-1)

        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = int(np.round(self.fg_ratio * self.roi_per_sample))
        fg_thresh = min(self.reg_pos_iou_range[0], self.cls_pos_iou_range[0])
        fg_inds = np.nonzero(max_overlaps >= fg_thresh)[0]
        fg_inds = np.concatenate(
            (fg_inds, roi_assignment), axis=0
        )  # consider the roi which has max_overlaps with gt as fg

        easy_bg_inds = np.nonzero((max_overlaps < self.cls_neg_iou_range[0]))[0]
        hard_bg_inds = np.nonzero(
            (max_overlaps < self.cls_neg_iou_range[1])
            & (max_overlaps >= self.cls_neg_iou_range[0])
        )[0]

        fg_num_rois = fg_inds.size
        bg_num_rois = hard_bg_inds.size + easy_bg_inds.size

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
            rand_num = np.random.permutation(fg_num_rois)
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = self.roi_per_sample - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(
                np.random.rand(self.roi_per_sample) * fg_num_rois
            ).astype(np.int32)
            fg_inds = fg_inds[rand_num]
            fg_rois_per_this_image = self.roi_per_sample
            bg_rois_per_this_image = 0
        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = self.roi_per_sample
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image
            )
            fg_rois_per_this_image = 0
        else:
            import pdb

            pdb.set_trace()
            raise NotImplementedError

        # augment the rois by noise
        roi_list, roi_iou_list, roi_gt_list = [], [], []
        if fg_rois_per_this_image > 0:
            fg_rois_src = roi_boxes3d[fg_inds].copy()
            gt_of_fg_rois = gt_info[gt_assignment[fg_inds]]
            fg_rois, fg_iou3d = self.aug_roi_by_noise(
                fg_rois_src, gt_of_fg_rois[:, :7], aug_times=10
            )
            roi_list.append(fg_rois)
            roi_iou_list.append(fg_iou3d)
            roi_gt_list.append(gt_of_fg_rois)

        if bg_rois_per_this_image > 0:
            bg_rois_src = roi_boxes3d[bg_inds].copy()
            gt_of_bg_rois = gt_info[gt_assignment[bg_inds]]
            bg_rois, bg_iou3d = self.aug_roi_by_noise(
                bg_rois_src, gt_of_bg_rois[:, :7], aug_times=1
            )
            roi_list.append(bg_rois)
            roi_iou_list.append(bg_iou3d)
            roi_gt_list.append(gt_of_bg_rois)

        rois = np.concatenate(roi_list, axis=0)
        iou_of_rois = np.concatenate(roi_iou_list, axis=0)
        gt_of_rois = np.concatenate(roi_gt_list, axis=0)

        return rois, iou_of_rois, gt_of_rois

    def sample_bg_inds(self, hard_bg_inds, easy_bg_inds, bg_rois_per_this_image):
        if hard_bg_inds.size > 0 and easy_bg_inds.size > 0:
            hard_bg_rois_num = int(bg_rois_per_this_image * self.hard_bg_ratio)
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_num = np.floor(
                np.random.rand(hard_bg_rois_num) * hard_bg_inds.size
            ).astype(np.int32)
            hard_bg_inds = hard_bg_inds[rand_num]
            # sampling easy bg
            rand_num = np.floor(
                np.random.rand(easy_bg_rois_num) * easy_bg_inds.size
            ).astype(np.int32)
            easy_bg_inds = easy_bg_inds[rand_num]

            bg_inds = np.concatenate([hard_bg_inds, easy_bg_inds], axis=0)
        elif hard_bg_inds.size > 0 and easy_bg_inds.size == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_num = np.floor(
                np.random.rand(hard_bg_rois_num) * hard_bg_inds.size
            ).astype(np.int32)
            bg_inds = hard_bg_inds[rand_num]
        elif hard_bg_inds.size == 0 and easy_bg_inds.size > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_num = np.floor(
                np.random.rand(easy_bg_rois_num) * easy_bg_inds.size
            ).astype(np.int32)
            bg_inds = easy_bg_inds[rand_num]
        else:
            raise NotImplementedError

        return bg_inds

    def aug_roi_by_noise(self, roi_boxes3d, gt_boxes3d, aug_times=10):
        """
        :param roi_boxes3d: (N, 7)
        :param gt_boxes3d: (N, 7)
        :return:
        """
        iou_of_rois = np.zeros(roi_boxes3d.shape[0], dtype=np.float32)
        for k in range(roi_boxes3d.__len__()):
            temp_iou = cnt = 0
            roi_box3d = roi_boxes3d[k]
            gt_box3d = gt_boxes3d[k]
            pos_thresh = min(self.reg_pos_iou_range[0], self.cls_pos_iou_range[0])
            gt_corners = box_8c_encoder.np_box_3d_to_box_8co(gt_box3d.reshape(1, 7))
            aug_box3d = roi_box3d
            while temp_iou < pos_thresh and cnt < aug_times:
                if np.random.rand() < 0.2:
                    aug_box3d = roi_box3d  # p=0.2 to keep the original roi box
                else:
                    aug_box3d = self.random_aug_box3d(roi_box3d)
                aug_corners = box_8c_encoder.np_box_3d_to_box_8co(
                    aug_box3d.reshape(1, 7)
                )
                temp_iou, _ = box_util.box3d_iou(
                    aug_corners[0, :, :], gt_corners[0, :, :]
                )
                cnt += 1
            roi_boxes3d[k] = aug_box3d
            iou_of_rois[k] = temp_iou
        return roi_boxes3d, iou_of_rois

    def random_aug_box3d(self, box3d):
        """
        :param box3d: (7) [x, y, z, h, w, l, ry]
        random shift, scale, orientation
        """
        if self.reg_aug_method == "single":
            pos_shift = np.random.rand(3) - 0.5  # [-0.5 ~ 0.5]
            hwl_scale = (np.random.rand(3) - 0.5) / (0.5 / 0.15) + 1.0  #
            angle_rot = (np.random.rand(1) - 0.5) / (
                0.5 / (np.pi / 12)
            )  # [-pi/12 ~ pi/12]

            aug_box3d = np.concatenate(
                [box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale, box3d[6:7] + angle_rot]
            )
            return aug_box3d
        elif self.reg_aug_method == "multiple":
            # pos_range, hwl_range, angle_range, mean_iou
            range_config = [
                [0.2, 0.1, np.pi / 12, 0.7],
                [0.3, 0.15, np.pi / 12, 0.6],
                [0.5, 0.15, np.pi / 9, 0.5],
                [0.8, 0.15, np.pi / 6, 0.3],
                [1.0, 0.15, np.pi / 3, 0.2],
            ]
            idx = np.random.randint(len(range_config))

            pos_shift = ((np.random.rand(3) - 0.5) / 0.5) * range_config[idx][0]
            hwl_scale = ((np.random.rand(3) - 0.5) / 0.5) * range_config[idx][1] + 1.0
            angle_rot = ((np.random.rand(1) - 0.5) / 0.5) * range_config[idx][2]

            aug_box3d = np.concatenate(
                [box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale, box3d[6:7] + angle_rot]
            )
            return aug_box3d
        elif self.reg_aug_method == "normal":
            x_shift = np.random.normal(loc=0, scale=0.3)
            y_shift = np.random.normal(loc=0, scale=0.2)
            z_shift = np.random.normal(loc=0, scale=0.3)
            h_shift = np.random.normal(loc=0, scale=0.25)
            w_shift = np.random.normal(loc=0, scale=0.15)
            l_shift = np.random.normal(loc=0, scale=0.5)
            ry_shift = ((np.random.rand() - 0.5) / 0.5) * np.pi / 12

            aug_box3d = np.array(
                [
                    box3d[0] + x_shift,
                    box3d[1] + y_shift,
                    box3d[2] + z_shift,
                    box3d[3] + h_shift,
                    box3d[4] + w_shift,
                    box3d[5] + l_shift,
                    box3d[6] + ry_shift,
                ]
            )
            return aug_box3d
        else:
            raise NotImplementedError

    def _shuffle_samples(self):
        perm = np.arange(self.num_samples)
        np.random.shuffle(perm)
        self.sample_list = self.sample_list[perm]

    def next_batch(self, batch_size, shuffle, **kwargs):
        """
        Retrieve the next `batch_size` samples from this data set.

        Args:
            batch_size: number of samples in the batch
            shuffle: whether to shuffle the indices after an epoch is completed

        Returns:
            list of dictionaries containing sample information
        """

        # Create empty set of samples
        samples_in_batch = []

        start = self._index_in_epoch
        # Shuffle only for the first epoch
        if self.epochs_completed == 0 and start == 0 and shuffle:
            self._shuffle_samples()

        while len(samples_in_batch) < batch_size:
            remain = batch_size - len(samples_in_batch)
            # Go to the next epoch
            start = self._index_in_epoch
            if start + remain >= self.num_samples:

                # Finished epoch
                self.epochs_completed += 1

                # Get the rest examples in this epoch
                rest_num_examples = self.num_samples - start

                # Append those samples to the current batch
                samples_in_batch.extend(
                    self.load_samples(np.arange(start, self.num_samples), **kwargs)
                )

                # Shuffle the data
                if shuffle:
                    self._shuffle_samples()

                # Start next epoch
                start = 0
                self._index_in_epoch = remain - rest_num_examples
                end = self._index_in_epoch

                # Append the rest of the batch
                samples_in_batch.extend(
                    self.load_samples(np.arange(start, end), **kwargs)
                )

            else:
                self._index_in_epoch += remain
                end = self._index_in_epoch

                # Append the samples in the range to the batch
                samples_in_batch.extend(
                    self.load_samples(np.arange(start, end), **kwargs)
                )

        return self.collate_batch(samples_in_batch)

    def collate_batch(self, samples):

        batch_size = samples.__len__()
        batch_data = {}
        sample_names = []

        for key in samples[0].keys():
            if key == constants.KEY_SAMPLE_NAME:
                sample_names = [samples[k][key] for k in range(batch_size)]
                continue
            if key == constants.KEY_SAMPLE_AUGS:
                continue

            if key == constants.KEY_LABEL_BOXES_3D:
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, samples[k][key].__len__())
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
                for i in range(batch_size):
                    batch_gt_boxes3d[i, : samples[i][key].__len__(), :] = samples[i][
                        key
                    ]
                batch_data[key] = batch_gt_boxes3d
                continue

            if isinstance(samples[0][key], np.ndarray):
                if batch_size == 1:
                    batch_data[key] = samples[0][key][np.newaxis, ...]
                else:
                    batch_data[key] = np.concatenate(
                        [samples[k][key][np.newaxis, ...] for k in range(batch_size)],
                        axis=0,
                    )
            else:
                batch_data[key] = [samples[k][key] for k in range(batch_size)]
                if isinstance(samples[0][key], int):
                    batch_data[key] = np.array(batch_data[key], dtype=np.int32)
                elif isinstance(samples[0][key], float):
                    batch_data[key] = np.array(batch_data[key], dtype=np.float32)

        return batch_data, sample_names
