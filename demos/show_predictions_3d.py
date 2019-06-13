import os
import sys
import time

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
import mayavi.mlab as mlab
from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.obj_detection import evaluation

import avod
from avod.builders.dataset_builder import DatasetBuilder
from avod.core import box_3d_encoder
from avod.core import box_3d_projector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "mayavi"))  # viz_util
from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d


BOX_COLOUR_SCHEME = {
    "Car": "#00FF00",  # Green
    "Pedestrian": "#00FFFF",  # Teal
    "Cyclist": "#FFFF00",  # Yellow
}


def main():
    """This demo shows RPN proposals and AVOD predictions in 3D
    and 2D in image space. Given certain thresholds for proposals
    and predictions, it selects and draws the bounding boxes on
    the image sample. It goes through the entire proposal and
    prediction samples for the given dataset split.

    The proposals, overlaid, and prediction images can be toggled on or off
    separately in the options section.
    The prediction score and IoU with ground truth can be toggled on or off
    as well, shown as (score, IoU) above the detection.
    """
    dataset_config = DatasetBuilder.copy_config(DatasetBuilder.KITTI_VAL)

    ##############################
    # Options
    ##############################
    dataset_config.data_split = "val"

    rpn_score_threshold = 0.1
    avod_score_threshold = 0.1

    gt_classes = ["Car"]
    # gt_classes = ['Pedestrian', 'Cyclist']

    # Overwrite this to select a specific checkpoint
    global_step = None
    checkpoint_name = "rpn_cars_fuse"

    # Drawing Toggles
    draw_proposals_separate = False
    draw_rpn_feature = False
    draw_overlaid = False
    draw_predictions_separate = True

    # Show orientation for both GT and proposals/predictions
    draw_orientations_on_prop = True
    draw_orientations_on_pred = True

    ##############################
    # End of Options
    ##############################

    # Get the dataset
    dataset = DatasetBuilder.build_kitti_dataset(dataset_config)

    # Setup Paths
    predictions_dir = (
        # avod.root_dir() + "/data/outputs/" + checkpoint_name + "/predictions_for_rcnn_train"
        # avod.root_dir() + "/data/outputs/" + checkpoint_name + "/predictions_for_rcnn_eval"
        avod.root_dir()
        + "/data/outputs/"
        + checkpoint_name
        + "/predictions"
    )

    proposals_and_scores_dir = (
        predictions_dir + "/proposals_and_scores/" + dataset.data_split
    )

    rpn_feature_dir = predictions_dir + "/rpn_feature/" + dataset.data_split

    predictions_and_scores_dir = (
        predictions_dir + "/final_predictions_and_scores/" + dataset.data_split
    )

    # Output images directories
    output_dir_base = predictions_dir + "/vis_3d"

    # Get checkpoint step
    steps = os.listdir(
        proposals_and_scores_dir
        if os.path.exists(proposals_and_scores_dir)
        else predictions_and_scores_dir
    )
    steps.sort(key=int)
    print("Available steps: {}".format(steps))

    # Use latest checkpoint if no index provided
    if global_step is None:
        global_step = steps[-1]

    if draw_proposals_separate:
        prop_out_dir = output_dir_base + "/proposals/{}/{}/{}".format(
            dataset.data_split, global_step, rpn_score_threshold
        )

        if not os.path.exists(prop_out_dir):
            os.makedirs(prop_out_dir)

        print("Proposal vis saved to:", prop_out_dir)

        if draw_rpn_feature:
            rpn_feature_out_dir = output_dir_base + "/rpn_feature/{}/{}/{}".format(
                dataset.data_split, global_step, rpn_score_threshold
            )

            if not os.path.exists(rpn_feature_out_dir):
                os.makedirs(rpn_feature_out_dir)

            print("RPN feature vis saved to:", rpn_feature_out_dir)

    if draw_overlaid:
        overlaid_out_dir = output_dir_base + "/overlaid/{}/{}/{}".format(
            dataset.data_split, global_step, avod_score_threshold
        )

        if not os.path.exists(overlaid_out_dir):
            os.makedirs(overlaid_out_dir)

        print("Overlaid images saved to:", overlaid_out_dir)

    if draw_predictions_separate:
        pred_out_dir = output_dir_base + "/predictions/{}/{}/{}".format(
            dataset.data_split, global_step, avod_score_threshold
        )

        if not os.path.exists(pred_out_dir):
            os.makedirs(pred_out_dir)

        print("Prediction vis saved to:", pred_out_dir)

    # Rolling average array of times for time estimation
    avg_time_arr_length = 10
    last_times = np.repeat(time.time(), avg_time_arr_length) + np.arange(
        avg_time_arr_length
    )

    for sample_idx in range(dataset.num_samples):
        # Estimate time remaining with 5 slowest times
        start_time = time.time()
        last_times = np.roll(last_times, -1)
        last_times[-1] = start_time
        avg_time = np.mean(np.sort(np.diff(last_times))[-5:])
        samples_remaining = dataset.num_samples - sample_idx
        est_time_left = avg_time * samples_remaining

        # Print progress and time remaining estimate
        sys.stdout.write(
            "\r\nSaving {} / {}, Avg Time: {:.3f}s, "
            "Time Remaining: {:.2f}s\n".format(
                sample_idx + 1, dataset.num_samples, avg_time, est_time_left
            )
        )
        sys.stdout.flush()

        sample_name = dataset.sample_names[sample_idx]
        img_idx = int(sample_name)

        ##############################
        # Proposals
        ##############################
        if draw_proposals_separate or draw_overlaid:
            # Load proposals from files
            proposals_file_path = proposals_and_scores_dir + "/{}/{}.txt".format(
                global_step, sample_name
            )
            if not os.path.exists(proposals_file_path):
                print("\tSample {}: No proposals, skipping".format(sample_name))
                continue

            proposals_and_scores = np.unique(
                np.loadtxt(proposals_file_path).reshape(-1, 8), axis=0
            )

            proposal_boxes_3d = proposals_and_scores[:, 0:7]
            proposal_scores = proposals_and_scores[:, 7]

            # Apply score mask to proposals
            score_mask = proposal_scores > rpn_score_threshold
            proposal_boxes_3d = proposal_boxes_3d[score_mask]
            proposal_scores = proposal_scores[score_mask]

            proposal_objs = [
                box_3d_encoder.box_3d_to_object_label(proposal, obj_type="Proposal")
                for proposal in proposal_boxes_3d
            ]
            if draw_rpn_feature:
                rpn_feature_path = rpn_feature_dir + "/{}/{}.npy".format(
                    global_step, sample_name
                )
                rpn_features = np.load(rpn_feature_path)
                rpn_pts, rpn_intensity, rpn_fg_mask, rpn_fts = (
                    rpn_features[:, 0:3],
                    rpn_features[:, 3],
                    rpn_features[:, 4],
                    rpn_features[:, 5:],
                )

        ##############################
        # Predictions
        ##############################
        if draw_predictions_separate or draw_overlaid:
            predictions_file_path = predictions_and_scores_dir + "/{}/{}.txt".format(
                global_step, sample_name
            )
            if not os.path.exists(predictions_file_path):
                print("\tSample {}: No predictions, skipping".format(sample_name))
                continue

            # Load predictions from files
            predictions_and_scores = np.loadtxt(
                predictions_and_scores_dir
                + "/{}/{}.txt".format(global_step, sample_name)
            ).reshape(-1, 9)

            prediction_boxes_3d = predictions_and_scores[:, 0:7]
            prediction_scores = predictions_and_scores[:, 7]
            prediction_class_indices = predictions_and_scores[:, 8]

            # process predictions only if we have any predictions left after
            # masking
            if len(prediction_boxes_3d) > 0:

                # Apply score mask
                avod_score_mask = prediction_scores >= avod_score_threshold
                prediction_boxes_3d = prediction_boxes_3d[avod_score_mask]
                prediction_scores = prediction_scores[avod_score_mask]
                prediction_class_indices = prediction_class_indices[avod_score_mask]

                # # Swap l, w for predictions where w > l
                # swapped_indices = \
                #     prediction_boxes_3d[:, 4] > prediction_boxes_3d[:, 3]
                # prediction_boxes_3d = np.copy(prediction_boxes_3d)
                # prediction_boxes_3d[swapped_indices, 3] = \
                #     prediction_boxes_3d[swapped_indices, 4]
                # prediction_boxes_3d[swapped_indices, 4] = \
                #     prediction_boxes_3d[swapped_indices, 3]

            # Convert to objs
            prediction_objs = [
                box_3d_encoder.box_3d_to_object_label(prediction, obj_type="Prediction")
                for prediction in prediction_boxes_3d
            ]
            for (obj, score) in zip(prediction_objs, prediction_scores):
                obj.score = score

        ##############################
        # Ground Truth
        ##############################

        # Get ground truth labels
        if dataset.has_labels:
            gt_objects = obj_utils.read_labels(dataset.label_dir, img_idx)
        else:
            gt_objects = []

        # Filter objects to desired difficulty
        filtered_gt_objs = dataset.kitti_utils.filter_labels(
            gt_objects, classes=gt_classes
        )

        image_path = dataset.get_rgb_image_path(sample_name)
        image = Image.open(image_path)
        image_size = image.size
        pts_rect, pts_intensity = dataset.kitti_utils.get_point_cloud(
            int(sample_name), (image_size[1], image_size[0])
        )

        ##############################
        # Reformat and prepare to draw
        ##############################
        if draw_proposals_separate or draw_overlaid:
            print("\tSample {}: Drawing proposals".format(sample_name))
            fig = draw_3d_predictions(
                pts_rect, filtered_gt_objs, proposal_objs, draw_orientations_on_prop
            )

            # Save just the proposals
            filename = prop_out_dir + "/" + sample_name + ".png"
            mlab.savefig(filename, figure=fig)

            if draw_rpn_feature:
                print("\tSample {}: Drawing rpn feature{:seg_mask}".format(sample_name))
                fig = draw_rpn_feature(rpn_pts, rpn_fg_mask)
                # Save just the proposals
                filename = rpn_feature_out_dir + "/" + sample_name + ".png"
                mlab.savefig(filename, figure=fig)

        if draw_overlaid or draw_predictions_separate:
            print("\tSample {}: Drawing predictions".format(sample_name))
            if draw_overlaid:
                # Overlay prediction boxes on image
                fig = draw_3d_predictions(
                    pts_rect,
                    filtered_gt_objs,
                    prediction_objs,
                    draw_orientations_on_pred,
                )
                filename = overlaid_out_dir + "/" + sample_name + ".png"
                mlab.savefig(filename, figure=fig)

            if draw_predictions_separate:
                # Now only draw prediction boxes on images
                # on a new figure handler
                fig = draw_3d_predictions(
                    pts_rect,
                    filtered_gt_objs,
                    prediction_objs,
                    draw_orientations_on_pred,
                )
                filename = pred_out_dir + "/" + sample_name + ".png"
                mlab.savefig(filename, figure=fig)
        mlab.close(all=True)
    print("\nDone")


def draw_3d_predictions(
    pts_rect, filtered_gt_objs, proposal_objs, draw_orientations_on_prop
):

    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in rect coord system) """

    print("\tFOV point num: {}".format(pts_rect.shape[0]))
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    draw_lidar(pts_rect, fig=fig)
    for obj in filtered_gt_objs:
        # Draw 3d bounding box
        box3d_pts_3d = obj_utils.compute_box_corners_3d(obj).T
        draw_gt_boxes3d([box3d_pts_3d], fig=fig, color=(1, 0, 0), draw_text=False)
        if draw_orientations_on_prop:
            # Draw heading arrow
            ori3d_pts_3d = obj_utils.compute_orientation_3d(obj).T
            x1, y1, z1 = ori3d_pts_3d[0, :]
            x2, y2, z2 = ori3d_pts_3d[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=(0.8, 0.5, 0.5),
                tube_radius=None,
                line_width=1,
                figure=fig,
            )
    for obj in proposal_objs:
        # Draw 3d bounding box
        box3d_pts_3d = obj_utils.compute_box_corners_3d(obj).T
        draw_gt_boxes3d([box3d_pts_3d], fig=fig, color=(0, 1, 0), draw_text=False)
        if draw_orientations_on_prop:
            # Draw heading arrow
            ori3d_pts_3d = obj_utils.compute_orientation_3d(obj).T
            x1, y1, z1 = ori3d_pts_3d[0, :]
            x2, y2, z2 = ori3d_pts_3d[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=(0.5, 0.5, 0.5),
                tube_radius=None,
                line_width=1,
                figure=fig,
            )
    mlab.show(1)
    input()
    return fig


def draw_rpn_feature(rpn_pts, rpn_fg_mask):
    print("\tRPN point num: ".format(rpn_pts.shape[0]))
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    draw_lidar(rpn_pts, color=rpn_pts[:, 2] + rpn_fg_mask * 100, fig=fig)
    mlab.show(1)
    input()
    return fig


def draw_prediction_info(
    ax,
    x,
    y,
    pred_obj,
    pred_class_idx,
    pred_box_2d,
    ground_truth,
    draw_score,
    draw_iou,
    gt_classes,
):

    label = ""

    if draw_score:
        label += "{:.2f}".format(pred_obj.score)

    if draw_iou and len(ground_truth) > 0:
        if draw_score:
            label += ", "
        iou = evaluation.two_d_iou(pred_box_2d, ground_truth)
        label += "{:.3f}".format(max(iou))

    box_cls = gt_classes[int(pred_class_idx)]

    ax.text(
        x,
        y - 4,
        gt_classes[int(pred_class_idx)] + "\n" + label,
        verticalalignment="bottom",
        horizontalalignment="center",
        color=BOX_COLOUR_SCHEME[box_cls],
        fontsize=10,
        fontweight="bold",
        path_effects=[patheffects.withStroke(linewidth=2, foreground="black")],
    )


if __name__ == "__main__":
    main()
