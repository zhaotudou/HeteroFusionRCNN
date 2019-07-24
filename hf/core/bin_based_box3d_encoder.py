"""
This module converts data to and from the 'box_3d' format
 [x, y, z, l, w, h, ry]
"""
import tensorflow as tf
import numpy as np


def tf_decode(
    ref_pts,
    ref_theta,
    bin_x,
    res_x_norm,
    bin_z,
    res_z_norm,
    bin_theta,
    res_theta_norm,
    res_y,
    res_size_norm,
    mean_sizes,
    Ss,
    DELTAs,
    R,
    DELTA_THETA,
):
    """Turns bin-based box3d format into an box_3d

    Input:
        ref_pts: (B,p,3) [x,y,z]
        ref_theta: (B,p) [ry] or a constant value
        
        bin_x: (B,p,K), bin assignments along X-axis
        res_x_norm: (B,p,K), normalized residual corresponds to bin_x
        bin_z: (B,p,K), bin assignments along Z-axis
        res_z_norm: (B,p,K), normalized residual corresponds to bin_z
        bin_theta: (B,p,K), bin assignments for orientation
        res_theta_norm: (B,p,K), normalized residual corresponds to bin_theta
        res_y: (B,p,K), residual w.r.t. ref_pts along Y-axis
        res_size_norm: (B,p,K,3), residual w.r.t. the average object size [l,w,h]

        mean_sizes, (B,p,K,3), average object size [l,w,h]
        Ss: XZ search range for different classes [-Ss, +Ss]
        DELTAs: XZ_BIN_LENs for different classes
        R: THETA search range [-R, +R]
        DELTA_THETA: THETA_BIN_LEN = 2 * R / NUM_BIN_THETA
    
    Output:
        boxes_3d: (B,p,K,7) 3D box in box_3d format [x, y, z, l, w, h, ry]
    """
    ndims = ref_pts.shape.ndims
    dx = (tf.to_float(bin_x) + 0.5) * DELTAs - Ss + res_x_norm * DELTAs
    dz = (tf.to_float(bin_z) + 0.5) * DELTAs - Ss + res_z_norm * DELTAs
    if ndims == 3:  # rpn
        K = tf.shape(bin_x)[2]
        if isinstance(ref_theta, tf.Tensor):
            # rotate along y
            all_rys = ref_theta
            ry_sin = tf.sin(all_rys)
            ry_cos = tf.cos(all_rys)
            rot_mats = tf.stack(
                [
                    tf.stack([ry_cos, ry_sin], axis=2),
                    tf.stack([-ry_sin, ry_cos], axis=2),
                ],
                axis=3,
            )
            rot_mats = tf.tile(tf.expand_dims(rot_mats, 2), [1, 1, K, 1, 1])
            dxz_rot = tf.matmul(
                rot_mats,
                tf.expand_dims(tf.stack([dx, dz], axis=3), axis=3),
                transpose_a=True,
                transpose_b=True,
            )
            dxz_rot = tf.squeeze(tf.matrix_transpose(dxz_rot), axis=3)
            dx = dxz_rot[:, :, :, 0]
            dz = dxz_rot[:, :, :, 1]
        else:
            assert ref_theta == 0
        ref_pts_tiled = tf.tile(tf.expand_dims(ref_pts, axis=2), [1, 1, K, 1])
        x = dx + ref_pts_tiled[:, :, :, 0]
        z = dz + ref_pts_tiled[:, :, :, 2]
        y = res_y + ref_pts_tiled[:, :, :, 1]
    elif ndims == 2:  # rcnn
        K = tf.shape(bin_x)[1]
        if isinstance(ref_theta, tf.Tensor):
            # rotate along y
            all_rys = ref_theta
            ry_sin = tf.sin(all_rys)
            ry_cos = tf.cos(all_rys)
            rot_mats = tf.stack(
                [
                    tf.stack([ry_cos, ry_sin], axis=1),
                    tf.stack([-ry_sin, ry_cos], axis=1),
                ],
                axis=2,
            )
            rot_mats = tf.tile(tf.expand_dims(rot_mats, 1), [1, K, 1, 1])
            dxz_rot = tf.matmul(
                rot_mats,
                tf.expand_dims(tf.stack([dx, dz], axis=2), axis=2),
                transpose_a=True,
                transpose_b=True,
            )
            dxz_rot = tf.squeeze(tf.matrix_transpose(dxz_rot), axis=2)
            dx = dxz_rot[:, :, 0]
            dz = dxz_rot[:, :, 1]
        else:
            assert ref_theta == 0
        ref_pts_tiled = tf.tile(tf.expand_dims(ref_pts, axis=1), [1, K, 1])
        x = dx + ref_pts_tiled[:, :, 0]
        z = dz + ref_pts_tiled[:, :, 2]
        y = res_y + ref_pts_tiled[:, :, 1]

        ref_theta = tf.tile(tf.expand_dims(ref_theta, axis=1), [1, K])

    theta = (
        ref_theta
        + (tf.to_float(bin_theta) + 0.5) * DELTA_THETA
        - R
        + res_theta_norm * 0.5 * DELTA_THETA
    )
    size = mean_sizes + res_size_norm * mean_sizes

    if ndims == 3:
        l = size[:, :, :, 0]
        w = size[:, :, :, 1]
        h = size[:, :, :, 2]
        # combine all
        boxes_3d = tf.stack([x, y, z, l, w, h, theta], axis=3)  # y+h/2
    elif ndims == 2:
        l = size[:, :, 0]
        w = size[:, :, 1]
        h = size[:, :, 2]
        # combine all
        boxes_3d = tf.stack([x, y, z, l, w, h, theta], axis=2)  # y+h/2

    return boxes_3d


def tf_encode(ref_pts, ref_theta, boxes_3d, mean_sizes, Ss, DELTAs, R, DELTA_THETA, K):
    """Turns box_3d into bin-based box3d format
    Input:
        ref_pts: (B,p,3) [x,y,z]
        ref_theta: (B,p) [ry] or a constant value
        boxes_3d: (B,p,7) 3D box in box_3d format [x, y, z, l, w, h, ry]
        
        mean_sizes, (B,p,3), average object size [l,w,h]
        Ss: XZ search range [-Ss, +Ss] for different classes
        DELTAs: XZ_BIN_LENs for different classes
        R: THETA search range [-R, +R]
        DELTA_THETA: THETA_BIN_LEN = 2 * R / NUM_BIN_THETA
    
    Output:
        bin_x: (B,p,K), bin assignments along X-axis
        res_x_norm: (B,p,K), normalized residual corresponds to bin_x
        bin_z: (B,p,K), bin assignments along Z-axis
        res_z_norm: (B,p,K), normalized residual corresponds to bin_z
        bin_theta: (B,p), bin assignments for orientation
        res_theta_norm: (B,p), normalized residual corresponds to bin_theta
        res_y: (B,p), residual w.r.t. ref_pts along Y-axis
        res_size_norm: (B,p,3), residual w.r.t. the average object size [l,w,h]
    """
    ndims = ref_pts.shape.ndims
    if ndims == 3:  # rpn
        dx = boxes_3d[:, :, 0] - ref_pts[:, :, 0]
        dy = boxes_3d[:, :, 1] - ref_pts[:, :, 1]  # - boxes_3d[:,:,5] / 2
        dz = boxes_3d[:, :, 2] - ref_pts[:, :, 2]
        if isinstance(ref_theta, tf.Tensor):
            # rotate along y
            all_rys = ref_theta * -1
            ry_sin = tf.sin(all_rys)
            ry_cos = tf.cos(all_rys)
            rot_mats = tf.stack(
                [
                    tf.stack([ry_cos, ry_sin], axis=2),
                    tf.stack([-ry_sin, ry_cos], axis=2),
                ],
                axis=3,
            )
            dxz_rot = tf.matmul(
                rot_mats,
                tf.expand_dims(tf.stack([dx, dz], axis=2), axis=2),
                transpose_a=True,
                transpose_b=True,
            )
            dxz_rot = tf.squeeze(tf.matrix_transpose(dxz_rot), axis=2)
            dx = dxz_rot[:, :, 0]
            dz = dxz_rot[:, :, 1]
        else:
            assert ref_theta == 0

        dx = tf.tile(tf.expand_dims(dx, -1), [1, 1, K])
        dz = tf.tile(tf.expand_dims(dz, -1), [1, 1, K])

        dsize = boxes_3d[:, :, 3:6] - mean_sizes

        dtheta = boxes_3d[:, :, 6] - ref_theta
        dtheta_shift = tf.clip_by_value(dtheta + R, 0.0, 2.0 * R - 1e-3)
    elif ndims == 2:  # rcnn
        dx = boxes_3d[:, 0] - ref_pts[:, 0]
        dy = boxes_3d[:, 1] - ref_pts[:, 1]  # - boxes_3d[:,5] / 2
        dz = boxes_3d[:, 2] - ref_pts[:, 2]
        if isinstance(ref_theta, tf.Tensor):
            # rotate along y
            all_rys = ref_theta * -1
            ry_sin = tf.sin(all_rys)
            ry_cos = tf.cos(all_rys)
            rot_mats = tf.stack(
                [
                    tf.stack([ry_cos, ry_sin], axis=1),
                    tf.stack([-ry_sin, ry_cos], axis=1),
                ],
                axis=2,
            )
            dxz_rot = tf.matmul(
                rot_mats,
                tf.expand_dims(tf.stack([dx, dz], axis=1), axis=1),
                transpose_a=True,
                transpose_b=True,
            )
            dxz_rot = tf.squeeze(tf.matrix_transpose(dxz_rot), axis=1)
            dx = dxz_rot[:, 0]
            dz = dxz_rot[:, 1]
        else:
            assert ref_theta == 0

        dx = tf.tile(tf.expand_dims(dx, -1), [1, K])
        dz = tf.tile(tf.expand_dims(dz, -1), [1, K])

        dsize = boxes_3d[:, 3:6] - mean_sizes

        dtheta = boxes_3d[:, 6] - tf.mod(ref_theta, 2 * np.pi)
        dtheta = tf.mod(dtheta, 2 * np.pi)
        dtheta = tf.where(
            tf.logical_and(
                tf.greater(dtheta, 0.5 * np.pi), tf.less(dtheta, 1.5 * np.pi)
            ),
            x=tf.mod(dtheta + np.pi, 2 * np.pi),
            y=dtheta,
        )
        dtheta_shift = tf.mod(dtheta + 0.5 * np.pi, 2 * np.pi)
        dtheta_shift = tf.clip_by_value(dtheta_shift - R, 1e-3, 2.0 * R - 1e-3)

    dx_shift = tf.clip_by_value(dx + Ss, 0.0, 2.0 * Ss - 1e-3)
    bin_x = tf.floor(dx_shift / DELTAs)
    res_x_norm = (dx_shift - (bin_x + 0.5) * DELTAs) / DELTAs

    dz_shift = tf.clip_by_value(dz + Ss, 0.0, 2.0 * Ss - 1e-3)
    bin_z = tf.floor(dz_shift / DELTAs)
    res_z_norm = (dz_shift - (bin_z + 0.5) * DELTAs) / DELTAs

    bin_theta = tf.floor(dtheta_shift / DELTA_THETA)
    res_theta_norm = (dtheta_shift - (bin_theta + 0.5) * DELTA_THETA) / (
        0.5 * DELTA_THETA
    )

    res_y = dy
    res_size_norm = dsize / mean_sizes

    return (
        tf.to_int32(bin_x),
        res_x_norm,
        tf.to_int32(bin_z),
        res_z_norm,
        tf.to_int32(bin_theta),
        res_theta_norm,
        res_y,
        res_size_norm,
    )
