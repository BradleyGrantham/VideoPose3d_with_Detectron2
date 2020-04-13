import json

import click
import numpy as np
import torch
from loguru import logger

import swing3d.constants
import swing3d.utils
import swing3d.pose3d
import swing3d.keypoints


@click.command()
@click.argument("input-video")
@click.option("--output-path")
@click.option(
    "--small-kps-model/--large-kps-model", default=True, help="Default is the small kps model."
)
def main(input_video, output_path, small_kps_model):
    # Initial pose predictor
    if small_kps_model:
        model_config = swing3d.constants.SMALL_MODEL_CONFIG_PATH
        model_path = swing3d.constants.SMALL_MODEL_WEIGHTS_PATH
    else:
        model_config = swing3d.constants.MODEL_CONFIG_PATH
        model_path = swing3d.constants.MODEL_WEIGHTS_PATH

    pose_predictor = swing3d.keypoints.init_pose_predictor(model_config, model_path, cuda=True)

    # Predict poses and save the result:
    img_generator = swing3d.utils.read_video(input_video)

    keypoints, resolution = swing3d.keypoints.predict_pose(pose_predictor, img_generator)

    ##############################################
    ############ VideoPose3D
    ##############################################

    # Normalize camera frame
    keypoints[..., :2] = swing3d.pose3d.normalize_screen_coordinates(
        keypoints[..., :2],
        w=resolution[1],
        h=resolution[0],
    )  # either 'custom' or 'detectron2'

    model_pos = swing3d.pose3d.TemporalModel(
        num_joints_in=keypoints.shape[-2],
        in_features=keypoints.shape[-1],
        num_joints_out=swing3d.constants.NUM_JOINTS,
        filter_widths=swing3d.constants.ARCHITECTURE,
        causal=swing3d.constants.CAUSAL,
        dropout=swing3d.constants.DROPOUT,
        channels=swing3d.constants.CHANNELS,
        dense=swing3d.constants.DENSE,
    )

    receptive_field = model_pos.receptive_field()
    logger.info(f"Receptive field: {receptive_field} frames")
    pad = (receptive_field - 1) // 2  # Padding on each side

    model_params = 0
    for parameter in model_pos.parameters():
        model_params += parameter.numel()
    logger.info(f"Trainable parameter count: {model_params}")

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()

    logger.info(f"Loading checkpoint {swing3d.constants.CHECKPOINT}")
    checkpoint = torch.load(
        swing3d.constants.CHECKPOINT, map_location=lambda storage, loc: storage
    )
    logger.info(f"This model was trained for {checkpoint['epoch']} epochs")
    model_pos.load_state_dict(checkpoint["model_pos"])

    # bgnote - pad with 121 on the first axis
    # bgnote - we will pass through our keypoints in one batch
    batch_2d = np.expand_dims(
        np.pad(keypoints, ((pad, pad), (0, 0), (0, 0)), "edge"), axis=0
    )

    prediction = swing3d.pose3d.evaluate(batch_2d, model_pos)

    prediction = swing3d.pose3d.camera_to_world(
        prediction, R=swing3d.constants.CAMERA_PARAMS["orientation"], t=0
    )

    prediction[:, :, 2] -= np.min(prediction[:, :, 2])

    prediction = swing3d.utils.rotate_about_z(prediction, np.pi / 2)
    prediction = swing3d.utils.rotate_about_y(prediction, np.pi)

    with open("/home/ubuntu/data.json", "w") as f:
        json.dump(prediction.tolist(), f)

    return prediction


if __name__ == "__main__":
    main()
