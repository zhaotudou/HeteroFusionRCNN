import numpy as np
import tensorflow as tf
from avod.core import compute_iou

# --------------------------------------
# Shared subgraphs for models
# --------------------------------------


def point_cloud_masking(mask, npoint=2048):
    """ Select point cloud with predicted 3D mask
    Input:
        mask: TF tensor in shape (B,P) of False (not pick) or True (pick)
        npoint: int scalar, maximum number of points to keep (default: 2048)
    Output:
        indices: TF tensor in shape (B,npoint,2)
    """
    mask = tf.to_float(mask)  # (B,P)

    def mask_to_indices(mask):
        indices = np.zeros((mask.shape[0], npoint, 2), dtype=np.int32)
        for i in range(mask.shape[0]):
            pos_indices = np.where(mask[i, :] > 0.5)[0]
            # skip cases when pos_indices is empty
            if len(pos_indices) > 0:
                if len(pos_indices) > npoint:
                    choice = np.random.choice(len(pos_indices), npoint, replace=False)
                else:
                    choice = np.random.choice(
                        len(pos_indices), npoint - len(pos_indices), replace=True
                    )
                    choice = np.concatenate((np.arange(len(pos_indices)), choice))
                np.random.shuffle(choice)
                indices[i, :, 1] = pos_indices[choice]
            indices[i, :, 0] = i
        return indices

    indices = tf.py_func(mask_to_indices, [mask], tf.int32)
    return indices


def foreground_masking(
    mask, num_fg_point, batch_size, pc_pts, pc_fts, seg_preds, seg_scores, label_box_3d
):
    """ Select foreground points and their features, segmentation scores etc. according to a given mask.
    Note the output number of foreground points is fixed by a given parameter. This is achived by sampling or padding.
    """
    fg_indices = point_cloud_masking(mask, num_fg_point)  # (B,F,2)
    foreground_pts = tf.reshape(
        tf.gather_nd(pc_pts, fg_indices),
        [batch_size, num_fg_point, pc_pts.shape[2].value],
    )  # (B,F,3)
    foreground_fts = tf.reshape(
        tf.gather_nd(pc_fts, fg_indices),
        [batch_size, num_fg_point, pc_fts.shape[2].value],
    )  # (B,F,C)
    foreground_preds = tf.reshape(
        tf.gather_nd(seg_preds, fg_indices), [batch_size, num_fg_point]
    )  # (B,F)
    foreground_scores = tf.reshape(
        tf.gather_nd(seg_scores, fg_indices), [batch_size, num_fg_point]
    )  # (B,F)
    foreground_label_boxes_3d = tf.reshape(
        tf.gather_nd(label_box_3d, fg_indices), [batch_size, num_fg_point, 7]
    )  # (B,F,7)
    return (
        foreground_pts,
        foreground_fts,
        foreground_preds,
        foreground_scores,
        foreground_label_boxes_3d,
    )


def gather_top_n(sb_data):
    sb_proposals, sb_confidences, sb_seg_scores, sb_sorted_idxs = sb_data
    sorted_confidences = tf.gather(sb_confidences, sb_sorted_idxs)
    sorted_proposals = tf.gather(sb_proposals, sb_sorted_idxs)
    sorted_seg_scores = tf.gather(sb_seg_scores, sb_sorted_idxs)
    return sorted_proposals, sorted_confidences, sorted_seg_scores


def sb_nms_fn(
    boxes_and_scores, oriented_nms, nms_iou_thresh, nms_size, fixed_num_proposal_nms
):
    """NMS on singel batch.
    Input:
        boxes_and_scores: A tuple with following elements:
            sb_boxes: (P, 7) [x, y, z, h, w, l, ry]
            sb_scores: (P). Scores for each 3d box
        oriented_nms: bool. Indicator for oriented NMS or non-oriented NMS
        nms_iou_thresh: float. The IoU threshold used for NMS
        nms_size: int. The number of output box indices.
        fixed_num_proposal_nms: bool. Only used when oriented_nms is set to True. For oriented NMS,
            if this is True, it is guaranteed that the number of boxes after NMS is fixed to nms_size,
            possibly with duplicated boxes. If this is False, the guarantee does not hold, but it is
            ceratin there will be no duplicated boxes. No matter what, the output of this funciton (the
            indices of selected boxes) will be of size of nms_size. This is achived by padding -1 if the
            number of selected boxes after NMS is less than nms_size.
    Output:
        sb_nms_indices_padded: (N). The indices of selected boxes, whose size is equal to nms_size. If the
            indice is -1, it means no selection and used just for padding.
        num_selected_boexs_before_padding: int. The number of selected boxes after NMS, i.e., before padding -1.
    """
    (sb_boxes, sb_scores) = boxes_and_scores
    # oriented_nms, nms_iou_thresh, nms_size, fixed_num_proposal_nms) = args
    if oriented_nms:
        sb_nms_indices = compute_iou.oriented_nms_tf(
            sb_boxes, sb_scores, nms_iou_thresh
        )

        sb_nms_indices = sb_nms_indices[:nms_size]
        if not fixed_num_proposal_nms:
            # sb_nms_indices append duplicated indices to make
            # sure sb_nms_indices has the same size as sb_boxes.
            # In case variable number proposals is desired,
            # use tf.unique to remove duplicates
            sb_nms_indices, _ = tf.unique(sb_nms_indices)

    else:
        sb_nms_indices = tf.image.non_max_suppression(
            sb_boxes, sb_scores, max_output_size=nms_size, iou_threshold=nms_iou_thresh
        )

    # sb_nms_indices = tf.Print(
    #     sb_nms_indices,
    #     [tf.shape(sb_nms_indices)[0]],
    #     "(avg) num_proposals_before_padding: ",
    # )

    sb_nms_indices_padded = tf.pad(
        sb_nms_indices,
        [[0, nms_size - tf.shape(sb_nms_indices)[0]]],
        mode="CONSTANT",
        constant_values=-1,
    )
    return sb_nms_indices_padded, tf.shape(sb_nms_indices)[0]


def sb_nms_selection(args):
    sb_proposals, sb_scores, sb_nms_indices = args
    sb_post_nms_proposals = tf.gather(sb_proposals, sb_nms_indices, axis=0)
    sb_post_nms_scores = tf.gather(sb_scores, sb_nms_indices, axis=0)
    return sb_post_nms_proposals, sb_post_nms_scores


def sb_compute_iou(args):
    proposal_boxes, gt_boxes = args
    return compute_iou.box3d_iou_tf(proposal_boxes, gt_boxes)


def x_z_theta_one_hot_encoding(bin_x, bin_z, bin_theta, S, DELTA, R, DELTA_THETA):
    bin_x_one_hot = tf.one_hot(
        tf.to_int32(bin_x), depth=int(2 * S / DELTA), on_value=1.0, off_value=0.0
    )

    bin_z_one_hot = tf.one_hot(
        tf.to_int32(bin_z), depth=int(2 * S / DELTA), on_value=1.0, off_value=0.0
    )

    bin_theta_one_hot = tf.one_hot(
        tf.to_int32(bin_theta),
        depth=int(2 * R / DELTA_THETA),
        on_value=1.0,
        off_value=0.0,
    )
    return bin_x_one_hot, bin_z_one_hot, bin_theta_one_hot
