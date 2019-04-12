import numpy as np
import tensorflow as tf

# --------------------------------------
# Shared subgraphs for models
# --------------------------------------

def point_cloud_masking(mask, npoint=2048):
    ''' Select point cloud with predicted 3D mask
    Input:
        mask: TF tensor in shape (B,P) of False (not pick) or True (pick)
        npoint: int scalar, maximum number of points to keep (default: 2048)
    Output:
        indices: TF tensor in shape (B,npoint,2)
    '''
    mask = tf.to_float(mask) #(B,P)
    
    def mask_to_indices(mask):
        indices = np.zeros((mask.shape[0], npoint, 2), dtype=np.int32)
        for i in range(mask.shape[0]):
            pos_indices = np.where(mask[i,:]>0.5)[0]
            # skip cases when pos_indices is empty
            if len(pos_indices) > 0:
                if len(pos_indices) > npoint:
                    choice = np.random.choice(len(pos_indices),
                        npoint, replace=False)
                else:
                    choice = np.random.choice(len(pos_indices),
                        npoint-len(pos_indices), replace=True)
                    choice = np.concatenate((np.arange(len(pos_indices)), choice))
                np.random.shuffle(choice)
                indices[i,:,1] = pos_indices[choice]
            indices[i,:,0] = i
        return indices

    indices = tf.py_func(mask_to_indices, [mask], tf.int32)
    return indices
