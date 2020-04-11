# KEYPOINT DETECTION
MODEL_CONFIG_PATH = (
    "../../detectron2/configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"
)
MODEL_WEIGHTS_PATH = "../../model_final_5ad38f.pkl"

SMALL_MODEL_CONFIG_PATH = (
    "../../detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
)
SMALL_MODEL_WEIGHTS_PATH = "../../model_final_a6e10b.pkl"

DEBUG_IMAGE_PATH = "../assets/debug_frame.png"
DEBUG_OUTPUT_PATH = "../assets/debug_output.png"

# POSE3D
CAMERA_PARAMS = {
    # Dummy camera parameters (taken from Human3.6M), only for visualization purposes
    "azimuth": 70,  # Only used for visualization
    "orientation": [
        0.1407056450843811,
        -0.1500701755285263,
        -0.755240797996521,
        0.6223280429840088,
    ],
    "translation": [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
}

NUM_JOINTS = 17
