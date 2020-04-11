import os
import time

import click
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model

import swing3d.constants
import swing3d.utils


def get_img_paths(imgs_dir):
    img_paths = []
    for dirpath, dirnames, filenames in os.walk(imgs_dir):
        for filename in [
            f
            for f in filenames
            if f.endswith(".png")
            or f.endswith(".PNG")
            or f.endswith(".jpg")
            or f.endswith(".JPG")
            or f.endswith(".jpeg")
            or f.endswith(".JPEG")
        ]:
            img_paths.append(os.path.join(dirpath, filename))
    img_paths.sort()

    return img_paths


def init_pose_predictor(config_path, weights_path, cuda=True):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = weights_path
    if cuda == False:
        cfg.MODEL.DEVICE = "cpu"
    # TODO - explore build_model here
    predictor = DefaultPredictor(cfg)

    return predictor


def encode_for_videpose3d(boxes, keypoints, resolution, dataset_name):
    # Generate metadata:
    metadata = {}
    metadata["layout_name"] = "coco"
    # bgnote - number of keypoints
    metadata["num_joints"] = swing3d.constants.NUM_JOINTS
    # bgnote - symmetrical keypoints (1,2 could be left hand, right hand for example)
    metadata["keypoints_symmetry"] = [
        [1, 3, 5, 7, 9, 11, 13, 15],
        [2, 4, 6, 8, 10, 12, 14, 16],
    ]
    metadata["video_metadata"] = {dataset_name: resolution}

    prepared_boxes = []
    prepared_keypoints = []
    for i in range(len(boxes)):
        if len(boxes[i]) == 0 or len(keypoints[i]) == 0:
            # No bbox/keypoints detected for this frame -> will be interpolated
            # bgnote - i feel we did this in the last step but shrug
            prepared_boxes.append(
                np.full(4, np.nan, dtype=np.float32)
            )  # 4 bounding box coordinates
            prepared_keypoints.append(
                np.full((swing3d.constants.NUM_JOINTS, 4), np.nan, dtype=np.float32)
            )  # 17 COCO keypoints
            continue

        prepared_boxes.append(boxes[i])
        # bgnote - i guess only :2 as we only need an x and y coord for each keypoint
        prepared_keypoints.append(keypoints[i][:, :2])

    boxes = np.array(prepared_boxes, dtype=np.float32)
    keypoints = np.array(prepared_keypoints, dtype=np.float32)
    keypoints = keypoints[:, :, :2]  # Extract (x, y)

    # Fix missing bboxes/keypoints by linear interpolation
    mask = ~np.isnan(boxes[:, 0])
    indices = np.arange(len(boxes))
    for i in range(4):
        boxes[:, i] = np.interp(indices, indices[mask], boxes[mask, i])
    for i in range(17):
        for j in range(2):
            keypoints[:, i, j] = np.interp(
                indices, indices[mask], keypoints[mask, i, j]
            )

    print("{} total frames processed".format(len(boxes)))
    print("{} frames were interpolated".format(np.sum(~mask)))
    print("----------")

    return (
        [
            {
                "start_frame": 0,  # Inclusive
                "end_frame": len(keypoints),  # Exclusive
                "bounding_boxes": boxes,
                "keypoints": keypoints,
            }
        ],
        metadata,
    )


def predict_pose(pose_predictor, img_generator, output_path, dataset_name="detectron2"):
    """
        pose_predictor: The detectron's pose predictor
        img_generator:  Images source
        output_path:    The path where the result will be saved in .npz format
    """
    boxes = []
    keypoints = []
    resolution = None

    # Predict poses:
    for i, img in enumerate(img_generator):
        pose_output = pose_predictor(img)

        if len(pose_output["instances"].pred_boxes.tensor) > 0:
            cls_boxes = pose_output["instances"].pred_boxes.tensor[0].cpu().numpy()
            cls_keyps = pose_output["instances"].pred_keypoints[0].cpu().numpy()
        else:
            cls_boxes = np.full((4,), np.nan, dtype=np.float32)
            cls_keyps = np.full(
                (17, 3), np.nan, dtype=np.float32
            )  # nan for images that do not contain human

        boxes.append(cls_boxes)
        keypoints.append(cls_keyps)

        # Set metadata:
        if resolution is None:
            resolution = {
                "w": img.shape[1],
                "h": img.shape[0],
            }

        print("{}      ".format(i + 1), end="\r")

    # Encode data in VidePose3d format and save it as a compressed numpy (.npz):
    data, metadata = encode_for_videpose3d(boxes, keypoints, resolution, dataset_name)
    output = {}
    output[dataset_name] = {}
    output[dataset_name]["custom"] = [data[0]["keypoints"].astype("float32")]

    return output, metadata


def save_keypoints(output, metadata, output_path):
    np.savez_compressed(output_path, positions_2d=output, metadata=metadata)


@click.command()
@click.argument("input-video")
@click.option("--output-path")
@click.option("--debug/--not-debug",)
def main(input_video, output_path, debug):
    start = time.time()
    # Initial pose predictor
    pose_predictor = init_pose_predictor(
        swing3d.constants.MODEL_CONFIG_PATH,
        swing3d.constants.MODEL_WEIGHTS_PATH,
        cuda=True,
    )

    # Predict poses and save the result:
    # img_generator = read_images('./images')    # read images from a directory

    if debug:
        imgs = cv2.imread(swing3d.constants.DEBUG_IMAGE_PATH)
        imgs = [cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)]
    else:
        imgs = swing3d.utils.read_video(input_video)  # or get them from a video

    if output_path is None:
        output_path = input_video.split("/")[-1].split(".")[0]

    keypoints, md = predict_pose(pose_predictor, imgs, output_path)

    if debug:
        swing3d.utils.plot_image_and_keypoints(
            imgs[0],
            np.squeeze(keypoints["detectron2"]["custom"][0]),
            output_path=swing3d.constants.DEBUG_OUTPUT_PATH,
        )
    else:
        save_keypoints(keypoints, md, output_path)

    print(f"Time taken: {time.time() - start}")


if __name__ == "__main__":
    main()
