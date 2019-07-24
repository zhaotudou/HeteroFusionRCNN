import copy
import numpy as np
import tensorflow as tf

from hf.core import obj_utils

COLOUR_SCHEME_PREDICTIONS = {
    "Easy GT": (255, 255, 0),  # Yellow
    "Medium GT": (255, 128, 0),  # Orange
    "Hard GT": (255, 0, 0),  # Red
    "Prediction": (50, 255, 50),  # Green
}


def get_gts_based_on_difficulty(dataset, img_idx):
    """Returns lists of ground-truth based on difficulty.
    """
    # Get all ground truth labels
    all_gt_objs = obj_utils.read_labels(dataset.label_dir, img_idx)

    # Filter to dataset classes
    gt_objs = dataset.kitti_utils.filter_labels(all_gt_objs)

    # Filter objects to desired difficulty
    easy_gt_objs = dataset.kitti_utils.filter_labels(
        copy.deepcopy(gt_objs), difficulty=0
    )
    medium_gt_objs = dataset.kitti_utils.filter_labels(
        copy.deepcopy(gt_objs), difficulty=1
    )
    hard_gt_objs = dataset.kitti_utils.filter_labels(
        copy.deepcopy(gt_objs), difficulty=2
    )

    for gt_obj in easy_gt_objs:
        gt_obj.type = "Easy GT"
    for gt_obj in medium_gt_objs:
        gt_obj.type = "Medium GT"
    for gt_obj in hard_gt_objs:
        gt_obj.type = "Hard GT"

    return easy_gt_objs, medium_gt_objs, hard_gt_objs, all_gt_objs


def two_d_iou(box, boxes):
    """Compute 2D IOU between a 2D bounding box 'box' and a list

    :param box: a numpy array in the form of [x1, y1, x2, y2] where (x1,y1) are
    image coordinates of the top-left corner of the bounding box, and (x2,y2)
    are the image coordinates of the bottom-right corner of the bounding box.

    :param boxes: a numpy array formed as a list of boxes in the form
    [[x1, y1, x2, y2], [x1, y1, x2, y2]].

    :return iou: a numpy array containing 2D IOUs between box and every element
    in numpy array boxes.
    """
    iou = np.zeros(len(boxes), np.float64)

    x1_int = np.maximum(box[0], boxes[:, 0])
    y1_int = np.maximum(box[1], boxes[:, 1])
    x2_int = np.minimum(box[2], boxes[:, 2])
    y2_int = np.minimum(box[3], boxes[:, 3])

    w_int = x2_int - x1_int
    h_int = y2_int - y1_int

    non_empty = np.logical_and(w_int > 0, h_int > 0)

    if non_empty.any():
        intersection_area = np.multiply(w_int[non_empty], h_int[non_empty])

        box_area = (box[2] - box[0]) * (box[3] - box[1])

        boxes_area = np.multiply(
            boxes[non_empty, 2] - boxes[non_empty, 0],
            boxes[non_empty, 3] - boxes[non_empty, 1],
        )

        union_area = box_area + boxes_area - intersection_area

        iou[non_empty] = intersection_area / union_area

    return iou.round(3)
