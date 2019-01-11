import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
cropping_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_cropping_so.so'))

def pc_crop_and_sample(pts, fts, boxes, box_ind, resize):
    '''
    input:
        pts:    (B, P, 3), [x, y, z],   float32
        fts:    (B, P, C), [c1,c2,...]  float32
        boxes:  (N, 6),    [cx,cy,cz,l,w,h] float32
        box_ind:(N)        [b]  int32
        resize = R, int32
    output:
        crop_pts:   (N, R, 3)   float32
        crop_fts:   (N, R, C)   float32
        crop_ind:   (N, R)      int32,  complementary to box_ind
        non_empty_box:  (N)         bool
    '''
    return cropping_module.pc_crop_and_sample(pts, fts, boxes, box_ind, resize)
## ops.NoGradient('PcCropAndSample')
@ops.RegisterGradient("PcCropAndSample")
def _PcCropAndSampleGrad(op, grad_crop_pts, grad_crop_fts, grad_crop_ind, grad_non_empty_box):
    # assume grad_crop_ind == None
    pts = op.inputs[0]
    fts = op.inputs[1]
    boxes = op.inputs[2]
    box_ind = op.inputs[3]
    
    crop_ind = op.outputs[2]

    #grad_pts = cropping_module.pc_crop_and_sample_grad_pts(pts, box_ind, crop_ind, grad_crop_pts)
    grad_fts = cropping_module.pc_crop_and_sample_grad_fts(fts, box_ind, crop_ind, grad_crop_fts)
    #grad_boxes = cropping_module.pc_crop_and_sample_grad_boxes(grad_crop_pts, grad_crop_fts)
    return [None, grad_fts, None, None]

if __name__=='__main__':


    batch_size = 1

    #np.random.seed(100)
    pts = np.random.rand(batch_size,100,3).astype('float32')
    fts = np.random.rand(batch_size,100,1).astype('float32')
    boxes = np.random.rand(2,6).astype('float32')
    box_ind = np.asarray([0, 0]).astype('int32')

    pts = tf.constant(pts)
    fts = tf.constant(fts)
    boxes = tf.constant(boxes)
    box_ind = tf.constant(box_ind)
    crop_pts, crop_fts, crop_ind, non_empty_box = pc_crop_and_sample(pts, fts, boxes, box_ind, 3)
    
    non_empty_fts = tf.boolean_mask(crop_fts, non_empty_box)

    with tf.Session() as sess:
        boxes, crop_pts, crop_fts, crop_ind, non_empty_box, non_empty_fts = sess.run(
        [boxes, crop_pts, crop_fts, crop_ind, non_empty_box, non_empty_fts])

    print("boxes:")
    print(boxes)
    print("crop_pts:")
    print(crop_pts)
    print("crop_fts:")
    print(crop_fts)
    print("crop_ind:")
    print(crop_ind)
    print("non_empty_box:")
    print(non_empty_box)
    print("non_empty_fts:")
    print(non_empty_fts)
