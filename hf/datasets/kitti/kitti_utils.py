import os

import numpy as np

from hf.core import obj_utils

from hf.core.label_cluster_utils import LabelClusterUtils
from hf.core.label_seg_utils import LabelSegUtils


class KittiUtils(object):
    # Definition for difficulty levels
    # These values are from Kitti dataset
    # 0 - easy, 1 - medium, 2 - hard
    HEIGHT = (40, 25, 25)
    OCCLUSION = (0, 1, 2)
    TRUNCATION = (0.15, 0.3, 0.5)

    def __init__(self, dataset):

        self.dataset = dataset

        # Label Clusters
        self.label_cluster_utils = LabelClusterUtils(self.dataset)
        self.clusters, self.std_devs = [None, None]
        # Label Seg Utils
        self.label_seg_utils = LabelSegUtils(self.dataset)
        self._label_seg_dir = self.label_seg_utils.label_seg_dir

        # Parse config
        self.config = dataset.config.kitti_utils_config
        self.area_extents = np.reshape(self.config.area_extents, (3, 2))
        self.bev_extents = self.area_extents[[0, 2]]
        self.expand_gt_size = self.config.label_seg_config.expand_gt_size

        # Label Clusters
        self.clusters, self.std_devs = self.label_cluster_utils.get_clusters()

    def class_str_to_index(self, class_str):
        """
        Converts an object class type string into a integer index

        Args:
            class_str: the object type (e.g. 'Car', 'Pedestrian', or 'Cyclist')

        Returns:
            The corresponding integer index for a class type, starting at 1
            (0 is reserved for the background class).
            Returns -1 if we don't care about that class type.
        """
        if class_str in self.dataset.classes:
            return self.dataset.classes.index(class_str) + 1

        raise ValueError(
            "Invalid class string {}, not in {}".format(class_str, self.dataset.classes)
        )

    def get_label_seg(self, classes_name, expand_gt_size, sample_name):

        label_seg = self.label_seg_utils.get_label_seg(
            classes_name, expand_gt_size, sample_name
        )
        return label_seg

    def get_point_cloud(self, img_idx, image_shape=None):
        """ Gets the points from the point cloud for a particular image,
            keeping only the points within the area extents

        Args:
            img_idx: An integer sample image index, e.g. 123 or 500
            image_shape: image dimensions (h, w), only required when
                source is 'lidar' or 'depth'

        Returns:
            points_rect: (N, 3). The set of points in rect camera coordinates
            points_intensity: (N, 1). The intensity values of the point
        """

        # wants im_size in (w, h) order
        im_size = [image_shape[1], image_shape[0]]

        point_cloud = obj_utils.get_lidar_point_cloud(
            img_idx, self.dataset.calib_dir, self.dataset.velo_dir, im_size=im_size
        )

        points_rect, points_intensity = (
            point_cloud[:, :-1],
            point_cloud[:, -1].reshape(-1, 1),
        )
        return points_rect, points_intensity

    def get_ground_plane(self, sample_name):
        """Reads the ground plane for the sample

        Args:
            sample_name: name of the sample, e.g. '000123'

        Returns:
            ground_plane: ground plane coefficients
        """
        ground_plane = obj_utils.get_road_plane(
            int(sample_name), self.dataset.planes_dir
        )
        return ground_plane

    def filter_labels(self, objects, classes=None, difficulty=None, max_occlusion=None):
        """Filters ground truth labels based on class, difficulty, and
        maximum occlusion

        Args:
            objects: A list of ground truth instances of Object Label
            classes: (optional) classes to filter by, if None
                all classes are used
            difficulty: (optional) KITTI difficulty rating as integer
            max_occlusion: (optional) maximum occlusion to filter objects

        Returns:
            filtered object label list
        """
        if classes is None:
            classes = self.dataset.classes

        objects = np.asanyarray(objects)
        filter_mask = np.ones(len(objects), dtype=np.bool)

        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]

            if filter_mask[obj_idx]:
                if not self._check_class(obj, classes):
                    filter_mask[obj_idx] = False
                    continue

            # Filter by difficulty (occlusion, truncation, and height)
            if difficulty is not None and not self._check_difficulty(obj, difficulty):
                filter_mask[obj_idx] = False
                continue

            if max_occlusion and obj.occlusion > max_occlusion:
                filter_mask[obj_idx] = False
                continue

        return objects[filter_mask]

    def _check_difficulty(self, obj, difficulty):
        """This filters an object by difficulty.
        Args:
            obj: An instance of ground-truth Object Label
            difficulty: An int defining the KITTI difficulty rate
        Returns: True or False depending on whether the object
            matches the difficulty criteria.
        """

        return (
            (obj.occlusion <= self.OCCLUSION[difficulty])
            and (obj.truncation <= self.TRUNCATION[difficulty])
            and (obj.y2 - obj.y1) >= self.HEIGHT[difficulty]
        )

    def _check_class(self, obj, classes):
        """This filters an object by class.
        Args:
            obj: An instance of ground-truth Object Label
        Returns: True or False depending on whether the object
            matches the desired class.
        """
        return obj.type in classes
