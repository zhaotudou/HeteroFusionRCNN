from __future__ import print_function

import tensorflow as tf
from bev_iou import bev_iou


def boxes3d_to_bev_tf(boxes3d):
    """
    Input:
        boxes3d: (N, 7) [x, y, z, l, w, h, ry]
    Output:
        boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
    """
    cu, cv = boxes3d[:, 0], boxes3d[:, 2]
    half_l, half_w = boxes3d[:, 3] / 2, boxes3d[:, 4] / 2
    x1, y1 = cu - half_l, cv - half_w
    x2, y2 = cu + half_l, cv + half_w
    ry = boxes3d[:, 6]
    boxes_bev = tf.stack([x1, y1, x2, y2, ry], axis=1)
    return boxes_bev


def box3d_iou_tf(boxes_a, boxes_b):
    """ Compute 3D bounding box IoU for Oriented BBox. Tensorflow version.

    Input:
        boxes_a: (N, 7) [x, y, z, h, w, l, ry]
        boxes_b: (M, 7) [x, y, z, h, w, l, ry]
    Output:
        iou_3d: (N, M) 3D bounding box IoU
        iou_2d: (N, M) bird's eye view 2D bounding box IoU
    """
    boxes_a_bev = boxes3d_to_bev_tf(boxes_a)
    boxes_b_bev = boxes3d_to_bev_tf(boxes_b)

    overlaps_bev, iou_2d = bev_iou.compute_bev_iou(boxes_a_bev, boxes_b_bev)

    # height overlap
    boxes_a_height_min = tf.reshape(boxes_a[:, 1] - boxes_a[:, 5], [-1, 1])
    boxes_a_height_min = tf.tile(boxes_a_height_min, [1, tf.shape(boxes_b)[0]])
    boxes_a_height_max = tf.reshape(boxes_a[:, 1], [-1, 1])
    boxes_a_height_max = tf.tile(boxes_a_height_max, [1, tf.shape(boxes_b)[0]])
    boxes_b_height_min = tf.reshape(boxes_b[:, 1] - boxes_b[:, 5], [1, -1])
    boxes_b_height_min = tf.tile(boxes_b_height_min, [tf.shape(boxes_a)[0], 1])
    boxes_b_height_max = tf.reshape(boxes_b[:, 1], [1, -1])
    boxes_b_height_max = tf.tile(boxes_b_height_max, [tf.shape(boxes_a)[0], 1])

    max_of_min = tf.maximum(boxes_a_height_min, boxes_b_height_min)
    min_of_max = tf.minimum(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = tf.clip_by_value(min_of_max - max_of_min, 0, tf.float32.max)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = tf.reshape(boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5], [-1, 1])
    vol_a = tf.tile(vol_a, [1, tf.shape(boxes_b)[0]])
    vol_b = tf.reshape(boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5], [1, -1])
    vol_b = tf.tile(vol_b, [tf.shape(boxes_a)[0], 1])

    iou_3d = overlaps_3d / tf.clip_by_value(
        vol_a + vol_b - overlaps_3d, 1e-7, tf.float32.max
    )

    return iou_3d, iou_2d


def oriented_nms_tf(boxes, scores, thresh):
    """
    Inputs:
        boxes: (N, 7) [x, y, z, h, w, l, ry]
        scores: (N)
        thresh: scalar. float
    Outputs:
        keep_idx: (N)
    """
    boxes_bev = boxes3d_to_bev_tf(boxes)
    _, sorted_idxs = tf.nn.top_k(scores, k=scores.shape[0])
    boxes_bev = tf.gather(boxes_bev, sorted_idxs)
    keep_idx = bev_iou.oriented_nms(boxes_bev, thresh)
    return tf.gather(sorted_idxs, keep_idx)


if __name__ == "__main__":

    boxes_a = tf.constant([[[1, 2, 3, 4, 5, 6, 7]]], dtype=tf.float32)
    boxes_b = tf.constant([[[1, 2, 3, 4, 5, 6, 7]]], dtype=tf.float32)

    def sb_compute_iou(args):
        proposal_boxes, gt_boxes = args
        return box3d_iou_tf(proposal_boxes, gt_boxes)

    iou3ds, iou2ds = tf.map_fn(
        sb_compute_iou, elems=(boxes_a, boxes_b), dtype=(tf.float32, tf.float32)
    )

    with tf.Session() as sess:
        iou3ds, iou2ds = sess.run([iou3ds, iou2ds])

    print("iou3ds:")
    print(iou3ds)
    print("iou2ds:")
    print(iou2ds)
