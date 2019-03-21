import numpy as np
import tensorflow as tf

def print_tensor(tensor_np):
    msg = 'contains_NaN' if np.isnan(tensor_np).any() else 'OK'
    msg += ' empty_size' if tensor_np.size == 0 else ' OK'
    print('==DEBUG==' + str(tensor_np.shape) + ' ' + msg)
    return tensor_np

def debug(tensor):
    debug_tensor = tf.py_func(print_tensor, [tensor], tensor.dtype)
    return debug_tensor

def debug_v2(tensor):
    print_op = tf.Print(tensor, [tensor], '==DEBUG==' + tensor.name + ' ')
    return print_op
