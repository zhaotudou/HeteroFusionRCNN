import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
interpolate_module = tf.load_op_library(os.path.join(BASE_DIR, "tf_interpolate_so.so"))


def three_nn(xyz1, xyz2):
    """
    Input:
        xyz1: (b,n,3) float32 array, unknown points
        xyz2: (b,m,3) float32 array, known points
    Output:
        dist: (b,n,3) float32 array, distances to known points
        idx: (b,n,3) int32 array, indices to known points
    """
    return interpolate_module.three_nn(xyz1, xyz2)


ops.NoGradient("ThreeNN")


def three_interpolate(points, idx, weight):
    """
    Input:
        points: (b,m,c) float32 array, known points
        idx: (b,n,3) int32 array, indices to known points
        weight: (b,n,3) float32 array, weights on known points
    Output:
        out: (b,n,c) float32 array, interpolated point values
    """
    points = tf.transpose(points, perm=(0, 2, 1))  # (b, c, m)
    ret = interpolate_module.three_interpolate(points, idx, weight)  # (b, c, n)
    return tf.transpose(ret, perm=(0, 2, 1))  # (b, n, c)


@tf.RegisterGradient("ThreeInterpolate")
def _three_interpolate_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    weight = op.inputs[2]
    return [
        interpolate_module.three_interpolate_grad(points, idx, weight, grad_out),
        None,
        None,
    ]


if __name__ == "__main__":
    import numpy as np
    import time

    # np.random.seed(100)
    pts = tf.random_normal((32, 128, 64))
    xyz1 = tf.random_normal((32, 512, 3))
    xyz2 = tf.random_normal((32, 128, 3))

    dist, idx = three_nn(xyz1, xyz2)
    # weight = tf.ones_like(dist) / 3.0
    dist = tf.maximum(dist, 1e-10)
    norm = tf.reduce_sum((1.0 / dist), axis=2, keep_dims=True)
    norm = tf.tile(norm, [1, 1, 3])
    weight = (1.0 / dist) / norm
    interpolated_points = three_interpolate(pts, idx, weight)
    with tf.Session() as sess:
        dist, idx, interpolated_points = sess.run([dist, idx, interpolated_points])
        # print(time.time() - now)
        print(interpolated_points)
        # print(ret.shape, ret.dtype)
        # print ret
