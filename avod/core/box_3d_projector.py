"""
Projects boxes into bird's eye view and image space.
Returns the 4 points (x, y) of the corresponding box
"""
import numpy as np
import tensorflow as tf

from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection import obj_utils

from avod.core import box_3d_encoder
from avod.core import box_8c_encoder
from avod.core import format_checker


def project_to_bev(boxes_3d, bev_extents):
    """
    Projects an array of 3D boxes into bird's eye view

    Args:
        boxes_3d: list of 3d boxes in the format:
            N x [x, y, z, l, w, h, ry]
        bev_extents: xz extents of the 3d area
            [[min_x, max_x], [min_z, max_z]]

    Returns:
        box_points: counter-clockwise order box points in bev map space
            N x [[x0, y0], ... [x3, y3]] - (N x 4 x 2)
        box_points_norm: points normalized as a percentage of the map size
            N x [[x0, y0], ... [x3, y3]] - (N x 4 x 2)
    """

    format_checker.check_box_3d_format(boxes_3d)

    boxes_3d = np.array(boxes_3d, dtype=np.float32)
    x = boxes_3d[:, 0]
    z = boxes_3d[:, 2]
    l = boxes_3d[:, 3]
    w = boxes_3d[:, 4]
    ry = boxes_3d[:, 6]

    # 1|0 2D corners
    # 2|3
    l_2 = l / 2.0
    w_2 = w / 2.0

    p0 = np.array([l_2, w_2])
    p1 = np.array([-l_2, w_2])
    p2 = np.array([-l_2, -w_2])
    p3 = np.array([l_2, -w_2])

    box_points = np.empty((len(boxes_3d), 4, 2))

    for box_idx in range(len(boxes_3d)):
        rot = ry[box_idx]

        rot_mat = np.reshape(
            [[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]], (2, 2)
        )

        box_x = x[box_idx]
        box_z = z[box_idx]

        box_xz = [box_x, box_z]

        box_p0 = np.dot(rot_mat, p0[:, box_idx]) + box_xz
        box_p1 = np.dot(rot_mat, p1[:, box_idx]) + box_xz
        box_p2 = np.dot(rot_mat, p2[:, box_idx]) + box_xz
        box_p3 = np.dot(rot_mat, p3[:, box_idx]) + box_xz

        box_points[box_idx] = np.array([box_p0, box_p1, box_p2, box_p3])

    # Calculate normalized box corners for ROI pooling
    x_extents_min = bev_extents[0][0]
    z_extents_min = bev_extents[1][1]  # z axis is reversed
    points_shifted = box_points - [x_extents_min, z_extents_min]

    x_extents_range = bev_extents[0][1] - bev_extents[0][0]
    z_extents_range = bev_extents[1][0] - bev_extents[1][1]
    box_points_norm = points_shifted / [x_extents_range, z_extents_range]

    box_points = np.asarray(box_points, dtype=np.float32)
    box_points_norm = np.asarray(box_points_norm, dtype=np.float32)

    return box_points, box_points_norm


def project_to_image_space(
    box_3d, calib_p2, truncate=False, image_size=None, discard_before_truncation=True
):
    """ Projects a box_3d into image space

    Args:
        box_3d: single box_3d to project
        calib_p2: stereo calibration p2 matrix
        truncate: if True, 2D projections are truncated to be inside the image
        image_size: [w, h] must be provided if truncate is True,
            used for truncation
        discard_before_truncation: If True, discard boxes that are larger than
            80% of the image in width OR height BEFORE truncation. If False,
            discard boxes that are larger than 80% of the width AND
            height AFTER truncation.

    Returns:
        Projected box in image space [x1, y1, x2, y2]
            Returns None if box is not inside the image
    """

    format_checker.check_box_3d_format(box_3d)

    obj_label = box_3d_encoder.box_3d_to_object_label(box_3d)
    corners_3d = obj_utils.compute_box_corners_3d(obj_label)

    projected = calib_utils.project_to_image(corners_3d, calib_p2)

    x1 = np.amin(projected[0])
    y1 = np.amin(projected[1])
    x2 = np.amax(projected[0])
    y2 = np.amax(projected[1])

    img_box = np.array([x1, y1, x2, y2])

    if truncate:
        if not image_size:
            raise ValueError("Image size must be provided")

        image_w = image_size[0]
        image_h = image_size[1]

        # Discard invalid boxes (outside image space)
        if (
            img_box[0] > image_w
            or img_box[1] > image_h
            or img_box[2] < 0
            or img_box[3] < 0
        ):
            return None

        # Discard boxes that are larger than 80% of the image width OR height
        if discard_before_truncation:
            img_box_w = img_box[2] - img_box[0]
            img_box_h = img_box[3] - img_box[1]
            if img_box_w > (image_w * 0.8) or img_box_h > (image_h * 0.8):
                return None

        # Truncate remaining boxes into image space
        if img_box[0] < 0:
            img_box[0] = 0
        if img_box[1] < 0:
            img_box[1] = 0
        if img_box[2] > image_w:
            img_box[2] = image_w
        if img_box[3] > image_h:
            img_box[3] = image_h

        # Discard boxes that are covering the the whole image after truncation
        if not discard_before_truncation:
            img_box_w = img_box[2] - img_box[0]
            img_box_h = img_box[3] - img_box[1]
            if img_box_w > (image_w * 0.8) and img_box_h > (image_h * 0.8):
                return None

    return img_box


def tf_project_to_image_space(boxes, calib, image_shape):
    """
    Projects 3D tensor boxes into image space

    Args:
        boxes: a tensor of anchors in the shape [B, 7].
            The anchors are in the format [x, y, z, l, h, w, ry]
        calib: tensor [3, 4] stereo camera calibration p2 matrix
        image_shape: a float32 tensor of shape [2]. This is dimension of
            the image [h, w]

    Returns:
        box_corners: a float32 tensor corners in image space -
            N x [x1, y1, x2, y2]
        box_corners_norm: a float32 tensor corners as a percentage
            of the image size - N x [x1, y1, x2, y2]
    """
    batch_size = boxes.shape[0]
    corners_3d = tf.matrix_transpose(
        box_8c_encoder.tf_box_3d_to_box_8co(boxes)
    )  # (B,8,3)
    corners_3d_hom = tf.concat(
        [corners_3d, tf.ones((batch_size, 8, 1))], axis=-1
    )  # (B,8,4)
    corners_3d_hom = tf.expand_dims(corners_3d_hom, axis=-1)  # (B,8,4,1)
    calib_tiled = tf.tile(tf.expand_dims(calib, 1), [1, 8, 1, 1])  # (B,8,3,4)
    projected_pts = tf.matmul(calib_tiled, corners_3d_hom)  # (B,8,3,1)
    projected_pts = tf.squeeze(projected_pts, axis=-1)  # (B,8,3)

    projected_pts_norm = projected_pts / tf.slice(
        projected_pts, [0, 0, 2], [-1, -1, 1]
    )  # divided by depth

    corners_2d = tf.gather(projected_pts_norm, [0, 1], axis=-1)  # (B,8,2)

    pts_2d_min = tf.reduce_min(corners_2d, axis=1)
    pts_2d_max = tf.reduce_max(corners_2d, axis=1)  # (B, 2)
    box_corners = tf.stack(
        [
            tf.gather(pts_2d_min, 0, axis=1),
            tf.gather(pts_2d_min, 1, axis=1),
            tf.gather(pts_2d_max, 0, axis=1),
            tf.gather(pts_2d_max, 1, axis=1),
        ],
        axis=1,
    )  # (B,4)

    # Normalize
    image_shape_h = image_shape[0]
    image_shape_w = image_shape[1]

    image_shape_tiled = tf.tile(
        [[image_shape_w, image_shape_h, image_shape_w, image_shape_h]], [batch_size, 1]
    )

    box_corners_norm = box_corners / tf.to_float(image_shape_tiled)

    return box_corners, box_corners_norm
