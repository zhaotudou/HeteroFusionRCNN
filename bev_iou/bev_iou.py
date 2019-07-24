import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
bev_iou_lib = tf.load_op_library(os.path.join(BASE_DIR, "bev_iou_so.so"))


def compute_bev_iou(proposals, gt_bboxes):
    """
    input:
        proposals: (N, 5), [x1, y1, x2, y2, ry], float32
        gt_bboxes: (M, 5), [x1, y1, x2, y2, ry]  float32
    output:
        overlap_area:   (N, M)   float32
        bev_iou:   (N, M)   float32
    """

    overlap_area, bev_iou = bev_iou_lib.compute_bev_iou(proposals, gt_bboxes)
    return overlap_area, bev_iou


ops.NoGradient("ComputeIOU3D")


def oriented_nms(boxes, thresh):
    """
    input:
        boxes: (N, 5), [x1, y1, x2, y2, ry], float32
    output:
        keep_idx: (N), int32
    """

    keep_idxs = bev_iou_lib.oriented_nms(boxes, nms_threshold=thresh)
    return keep_idxs


ops.NoGradient("OrientedNMS")


if __name__ == "__main__":

    proposals = np.asarray([[0, 0, 1, 1, 0], [2, 2, 3, 3, 0]], dtype=np.float32)
    gt = np.asarray(
        [[0, 0, 1, 1, 0], [2, 2, 4, 4, 0], [5, 5, 6, 6, 0]], dtype=np.float32
    )

    overlap_area, bev_iou = compute_bev_iou(proposals, gt)

    with tf.Session() as sess:
        overlap_area, bev_iou = sess.run([overlap_area, bev_iou])

    print("overlap_area:")
    print(overlap_area)
    print("\nbev_iou:")
    print(bev_iou)

    boxes = np.asarray(
        [[0, 0, 1, 1, 0], [2, 2, 3, 3, 0], [0, 0, 0.75, 0.75, 0]], dtype=np.float32
    )
    threshold = 0.5

    keep_idxs = oriented_nms(boxes, threshold)

    with tf.Session() as sess:
        keep_idxs = sess.run(keep_idxs)
    print("keep idxs:")
    print("keep idxs shape: ", keep_idxs.shape)
    print(keep_idxs)
