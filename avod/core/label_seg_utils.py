import os

import numpy as np
import tensorflow as tf

import avod

from avod.core.label_seg_preprocessor import LabelSegPreprocessor
from avod.core import box_8c_encoder


class LabelSegUtils:
    def __init__(self, dataset):

        self._dataset = dataset

        ##############################
        # Parse KittiUtils config
        ##############################
        self.kitti_utils_config = dataset.config.kitti_utils_config
        self._area_extents = self.kitti_utils_config.area_extents

        ##############################
        # Parse MiniBatchUtils config
        ##############################
        self.config = self.kitti_utils_config.label_seg_config
        self._expand_gt_size = self.config.expand_gt_size

        # Setup paths
        self.label_seg_dir = avod.root_dir() + '/data/label_segs/' + \
            dataset.name + '/' + dataset.cluster_split + '/' + \
            dataset.pc_source

    def preprocess_rpn_label_segs(self, indices):
        """Generates rpn mini batch info for the kitti dataset

            Preprocesses data and saves data to files.
            Each file contains information that is used to feed
            to the network for RPN training.
        """
        label_seg_preprocessor = \
            LabelSegPreprocessor(self._dataset,
                                  self.label_seg_dir,
                                  self._expand_gt_size)

        label_seg_preprocessor.preprocess(indices)
        
    def get_label_seg(self, classes_name, expand_gt_size, sample_name):
        """Reads in the file containing the information matrix

        Args:
            classes_name: object type, one of ('Car', 'Pedestrian',
                'Cyclist', 'People')
            sample_name: image name to read the corresponding file

        Returns:
            label_segs: class index of the point
                (e.g. 0 or 1, for "Background" or "Car")

            [] if the file contains an empty array
        """
        file_name = self.get_file_path(classes_name, expand_gt_size,
                                       sample_name)

        if not os.path.exists(file_name):
            raise FileNotFoundError(
                "{} not found for sample {} in {}, "
                "run the preprocessing script first".format(
                    file_name,
                    sample_name,
                    self.label_seg_dir))

        label_seg = np.load(file_name)
        return label_seg

    def get_file_path(self, classes_name, expand_gt_size, sample_name):
        """Gets the full file path to the anchors info

        Args:
            classes_name: name of classes ('Car', 'Pedestrian', 'Cyclist',
                'People')
            sample_name: sample name, e.g. '000123'

        Returns:
            The label seg file path. Returns the folder if
                sample_name is None
        """

        expand_gt_size = np.round(expand_gt_size, 3)
        if sample_name:
            return self.label_seg_dir + '/' + classes_name + \
                '[ ' + str(expand_gt_size) + ']/' + \
                sample_name + ".npy"

        return self.label_seg_dir + '/' + classes_name + \
            '[ ' + str(expand_gt_size) + ']'
    
    @classmethod 
    def label_point_cloud(cls, raw_xyz, bbox_8co, categories):
        '''
        Give all points a label if it is inside a box
         Input:
           raw_xyz: (N x 3)
           bbox_8co: (M x 8 x 3)
           categories: (M)
         Return:
           label_seg: (N)
        '''
        num_points = raw_xyz.shape[0]
        num_boxes = bbox_8co.shape[0]
        label_seg = np.zeros((num_points), dtype=int)
        if num_boxes > 0:
            facets = box_8c_encoder.np_box_8co_to_facet(bbox_8co)
        for i in range(num_boxes):
            category = categories[i]
            box = bbox_8co[i,:,:]
            facet = facets[i,:,:]
            x = box[:,0]
            y = box[:,1]
            z = box[:,2]
            max_x = x.max(axis = 0)
            max_y = y.max(axis = 0)
            max_z = z.max(axis = 0)
            min_x = x.min(axis = 0)
            min_y = y.min(axis = 0)
            min_z = z.min(axis = 0)
            
            for j in range(num_points):
                if label_seg[j] > 0:
                    continue
                point = raw_xyz[j,:]
                px, py, pz = point[0], point[1], point[2]
                # return false if point is siginificant away from all vertices
                if px > max_x or px < min_x or \
                   py > max_y or py < min_y or \
                   pz > max_z or pz < min_z:
                    continue
                elif cls.point_inside_facet(point, facet, box):
                    label_seg[j] = category
        return label_seg

    @classmethod
    def point_inside_facet(cls, point, facet, box):
        '''
        Check if point is inside an convex 3d object defined by facet
         Input:
           point: ([x,y,z])
           facet: (N x [a,b,c,d]) i.e. for a cube, N is 6
         Return:
           boolean
        '''
        for i in range(facet.shape[0]):
            norm = facet[i,0:3]
            A = facet[i,4:]
            D = point - A
            product = np.dot(norm, D)
            if product < 0:
              return False
        return True

def main():
    raw_xyz = np.asarray([[0.55, 0, 0.1],[-0.3, -0.5, -0.3]])
    bbox_3d = np.asarray([0, 0, 0, 1, 1, 1, 3.14/4])
    bbox_8co = box_8c_encoder.np_box_3d_to_box_8co(bbox_3d).T
    bbox_8co = bbox_8co.reshape(1, 8, 3)
    print(bbox_8co)
    categories = np.asarray([1])
    label_seg = LabelSegUtils.label_point_cloud(raw_xyz, bbox_8co, categories)
    print(label_seg)

if __name__ == '__main__':
    main()
