"""
This module converts data to and from the 'box_3d' format
 [x, y, z, l, w, h, ry]
"""
import tensorflow as tf

from wavedata.tools.obj_detection import obj_utils

def tf_decode(ref_pts,
              bin_x, res_x_norm, 
              bin_z, res_z_norm, 
              bin_theta, res_theta_norm, 
              res_y, res_size, mean_sizes,
              S, DELTA, R, DELTA_THETA):
    """Turns bin-based box3d format into an box_3d

    Input:
        ref_pts: (B,p,3) [x,y,z]
        
        bin_x: (B,p), bin assignments along X-axis
        res_x_norm: (B,p), normalized residual corresponds to bin_x 
        bin_z: (B,p), bin assignments along Z-axis
        res_z_norm: (B,p), normalized residual corresponds to bin_z
        bin_theta: (B,p), bin assignments for orientation
        res_theta_norm: (B,p), normalized residual corresponds to bin_theta
        res_y: (B,p), residual w.r.t. ref_pts along Y-axis
        res_size: (B,p,3), residual w.r.t. the average object size [l,w,h]

        mean_sizes, (B,p,3), average object size [l,w,h]
        S: XZ search range [-S, +S]
        DELTA: XZ_BIN_LEN
        R: THETA search range [-R, +R]
        DELTA_THETA: THETA_BIN_LEN = 2 * R / NUM_BIN_THETA
    
    Output:
        boxes_3d: (B,p,7) 3D box in box_3d format [x, y, z, l, w, h, ry]
    """

    x = tf.to_float(bin_x) * DELTA - S + ref_pts[:,:,0] + 0.5 * DELTA + res_x_norm * DELTA
    z = tf.to_float(bin_z) * DELTA - S + ref_pts[:,:,2] + 0.5 * DELTA + res_z_norm * DELTA
    theta = tf.to_float(bin_theta) * DELTA_THETA - R + 0.5 * DELTA_THETA + res_theta_norm * DELTA_THETA
    
    y = ref_pts[:,:,1] + res_y
    
    size = mean_sizes + res_size
    l = size[:,:,0]
    w = size[:,:,1]
    h = size[:,:,2]
   
    # combine all
    boxes_3d = tf.stack([x,y,z,l,w,h,theta], axis=2)
    
    return boxes_3d

def tf_encode(ref_pts, boxes_3d, mean_sizes, S, DELTA, R, DELTA_THETA):
    '''Turns box_3d into bin-based box3d format
    Input:
        ref_pts: (B,p,3) [x,y,z]
        boxes_3d: (B,p,7) 3D box in box_3d format [x, y, z, l, w, h, ry]
        
        mean_sizes, (B,p,3), average object size [l,w,h]
        S: XZ search range [-S, +S]
        DELTA: XZ_BIN_LEN
        R: THETA search range [-R, +R]
        DELTA_THETA: THETA_BIN_LEN = 2 * R / NUM_BIN_THETA
    
    Output:
        bin_x: (B,p), bin assignments along X-axis
        res_x_norm: (B,p), normalized residual corresponds to bin_x 
        bin_z: (B,p), bin assignments along Z-axis
        res_z_norm: (B,p), normalized residual corresponds to bin_z
        bin_theta: (B,p), bin assignments for orientation
        res_theta_norm: (B,p), normalized residual corresponds to bin_theta
        res_y: (B,p), residual w.r.t. ref_pts along Y-axis
        res_size: (B,p,3), residual w.r.t. the average object size [l,w,h]
    '''
    dx = boxes_3d[:,:,0] - ref_pts[:,:,0]
    dy = boxes_3d[:,:,1] - ref_pts[:,:,1]
    dz = boxes_3d[:,:,2] - ref_pts[:,:,2]
    dsize = boxes_3d[:,:,3:6] - mean_sizes 
    dtheta = boxes_3d[:,:,6]
    
    bin_x = tf.floor((dx + S) / DELTA)
    res_x_norm = (dx + S - (bin_x + 0.5) * DELTA) / DELTA
    
    bin_z = tf.floor((dz + S) / DELTA)
    res_z_norm = (dz + S - (bin_z + 0.5) * DELTA) / DELTA
    
    bin_theta = tf.floor((dtheta + R) / DELTA_THETA)
    res_theta_norm = (dtheta + R - (bin_theta + 0.5) * DELTA_THETA) / DELTA_THETA
    
    res_y = dy
    res_size = dsize

    return bin_x, res_x_norm, bin_z, res_z_norm, bin_theta, res_theta_norm, res_y, res_size

