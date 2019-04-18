import tensorflow as tf

def tf_rect_to_image(pts3d, calib):
    """
    Projects 3D points into image space

    Args:
        pts3d: a tensor of rect points in the shape [B, N, 3].
        calib: tensor [B,3,4] stereo camera calibration p2 matrix
    Returns:
        pts2d: a float32 tensor points in image space -
            B x N x [x, y]
    """
    B = pts3d.shape[0]
    N = pts3d.shape[1]
    calib_expand = tf.tile(tf.expand_dims(calib, 1), [1,N,1,1]) # (B,N,3, 4)
    pts3d_hom = tf.concat([pts3d, tf.ones((B,N,1))], axis=-1) # (B,N,4)
    pts3d_hom = tf.expand_dims(pts3d_hom, axis=-1) # (B,N,4,1)
    pts2d_hom = tf.matmul(calib_expand, pts3d_hom) # (B,N,3,1)
    pts2d_hom = tf.squeeze(pts2d_hom, axis=-1) # (B,N,3)
    depth = tf.gather(pts2d_hom, 2, axis=-1)
    return tf.stack([
        tf.gather(pts2d_hom, 0, axis=-1)/depth,
        tf.gather(pts2d_hom, 1, axis=-1)/depth,
    ], axis=-1)

