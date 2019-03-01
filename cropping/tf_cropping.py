import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import numpy as np
from avod.core import box_8c_encoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
cropping_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_cropping_so.so'))

def pc_crop_and_sample(pts, fts, mask, boxes, box_ind, resize):
    '''
    input:
        pts:    (B, P, 3), [x, y, z],   float32
        fts:    (B, P, C), [c1,c2,...]  float32
        mask:   (B, P) point mask label, bool
        boxes:  (N, 3, 8), 8 corners (x,y,z), float32
        box_ind:(N)        [b]  which batch that box belongs to, int32
        resize = R, int32
    output:
        crop_pts:   (N, R, 3)   float32
        crop_fts:   (N, R, C)   float32
        crop_mask:  (N, R)      bool
        crop_ind:   (N, R)      int32,  which point to crop, complementary to box_ind
        non_empty_box_mask:  (N)         bool
    '''
    return cropping_module.pc_crop_and_sample(pts, fts, mask, boxes, box_ind, resize)
## ops.NoGradient('PcCropAndSample')
@ops.RegisterGradient("PcCropAndSample")
def _PcCropAndSampleGrad(op, grad_crop_pts, grad_crop_fts, grad_crop_mask, grad_crop_ind, grad_non_empty_box):
    # assume grad_crop_ind == None
    pts = op.inputs[0]
    fts = op.inputs[1]
    boxes = op.inputs[2]
    box_ind = op.inputs[3]
    
    crop_ind = op.outputs[2]

    #grad_pts = cropping_module.pc_crop_and_sample_grad_pts(pts, box_ind, crop_ind, grad_crop_pts)
    grad_fts = cropping_module.pc_crop_and_sample_grad_fts(fts, box_ind, crop_ind, grad_crop_fts)
    #grad_boxes = cropping_module.pc_crop_and_sample_grad_boxes(grad_crop_pts, grad_crop_fts)
    return [None, grad_fts, None, None, None]

if __name__=='__main__':

    batch_size = 1
    pts = np.asarray([[[1.0, 0, 0.1],[-0.3, -0.5, -0.3]]], dtype=np.float32)
    fts = np.random.rand(batch_size,2,1).astype('float32')
    mask = np.random.rand(batch_size,2).astype('bool')
    boxes_3d = np.asarray([0, 0, 0, 1, 1, 1, 3.14/4], dtype=np.float32)
    boxes= box_8c_encoder.np_box_3d_to_box_8co(boxes_3d).reshape(-1,3,8).astype('float32')

    #np.random.seed(100)
    box_ind = np.asarray([0]).astype('int32')

    pts = tf.constant(pts)
    fts = tf.constant(fts)
    mask = tf.constant(mask)
    boxes = tf.constant(boxes)
    box_ind = tf.constant(box_ind)
    crop_pts, crop_fts, crop_mask, crop_ind, non_empty_box_mask = pc_crop_and_sample(pts, fts, mask, boxes, box_ind, 1)
    
    non_empty_fts = tf.boolean_mask(crop_fts, non_empty_box_mask)

    with tf.Session() as sess:
        crop_pts, crop_fts, crop_mask, crop_ind, non_empty_box_mask, non_empty_fts = sess.run(
        [crop_pts, crop_fts, crop_mask, crop_ind, non_empty_box_mask, non_empty_fts])

    print("crop_pts:")
    print(crop_pts)
    print("crop_fts:")
    print(crop_fts)
    print("crop_mask:")
    print(crop_mask)
    print("crop_ind:")
    print(crop_ind)
    print("non_empty_box:")
    print(non_empty_box_mask)
    print("non_empty_fts:")
    print(non_empty_fts)
