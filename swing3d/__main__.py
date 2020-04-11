import json

import click
import numpy as np
import torch

import swing3d.constants
import swing3d.utils
import swing3d.pose3d
from swing3d.keypoints import init_pose_predictor, predict_pose


@click.command()
@click.argument("input-video")
@click.option("--output-path")
@click.option(
    "--small-model/--large-model", default=True, help="Default is the small model."
)
def main(input_video, output_path, small_model):
    # Initial pose predictor
    if small_model:
        model_config = swing3d.constants.SMALL_MODEL_CONFIG_PATH
        model_path = swing3d.constants.SMALL_MODEL_WEIGHTS_PATH
    else:
        model_config = swing3d.constants.MODEL_CONFIG_PATH
        model_path = swing3d.constants.MODEL_WEIGHTS_PATH

    pose_predictor = init_pose_predictor(model_config, model_path, cuda=True)

    # Predict poses and save the result:
    img_generator = swing3d.utils.read_video(input_video)

    keypoints, keypoints_metadata = predict_pose(pose_predictor, img_generator)

    ##############################################
    ############ VideoPose3D
    ##############################################

    keypoints_symmetry = keypoints_metadata["keypoints_symmetry"]

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                kps[..., :2] = swing3d.pose3d.normalize_screen_coordinates(
                    kps[..., :2],
                    w=keypoints_metadata["video_metadata"]["detectron2"]["w"],
                    h=keypoints_metadata["video_metadata"]["detectron2"]["h"],
                )  # either 'custom' or 'detectron2'
                keypoints[subject][action][cam_idx] = kps

    poses_valid_2d = keypoints["detectron2"]["custom"]

    model_pos = swing3d.pose3d.TemporalModel(
        poses_valid_2d[0].shape[-2],
        poses_valid_2d[0].shape[-1],
        num_joints_out=swing3d.constants.NUM_JOINTS,
        filter_widths=swing3d.constants.ARCHITECTURE,
        causal=swing3d.constants.CAUSAL,
        dropout=swing3d.constants.DROPOUT,
        channels=swing3d.constants.CHANNELS,
        dense=swing3d.constants.DENSE,
    )

    receptive_field = model_pos.receptive_field()
    print("INFO: Receptive field: {} frames".format(receptive_field))
    pad = (receptive_field - 1) // 2  # Padding on each side
    causal_shift = 0

    model_params = 0
    for parameter in model_pos.parameters():
        model_params += parameter.numel()
    print("INFO: Trainable parameter count:", model_params)

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()

    print("Loading checkpoint", swing3d.constants.CHECKPOINT)
    checkpoint = torch.load(
        swing3d.constants.CHECKPOINT, map_location=lambda storage, loc: storage
    )
    print("This model was trained for {} epochs".format(checkpoint["epoch"]))
    model_pos.load_state_dict(checkpoint["model_pos"])

    # bgnote - pad with 121 on the first axis
    # bgnote - we will pass through our keypoints in one batch
    batch_2d = np.expand_dims(
        np.pad(poses_valid_2d[0], ((pad, pad), (0, 0), (0, 0)), "edge"), axis=0
    )

    prediction = swing3d.pose3d.evaluate(batch_2d, model_pos)

    prediction = swing3d.pose3d.camera_to_world(
        prediction, R=swing3d.constants.CAMERA_PARAMS["orientation"], t=0
    )

    prediction[:, :, 2] -= np.min(prediction[:, :, 2])

    prediction = swing3d.utils.rotate_about_z(prediction, np.pi / 2)
    prediction = swing3d.utils.rotate_about_y(prediction, np.pi)

    with open("data.json", "w") as f:
        json.dump(prediction.tolist(), f)

    return prediction


if __name__ == "__main__":
    main()
