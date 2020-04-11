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

JOINTS_LEFT = [4, 5, 6, 11, 12, 13],
JOINTS_RIGHT = [1, 2, 3, 14, 15, 16]

# POSE3D CLI ARGUMENTS
SUBJECTS_TRAIN = "S1,S5,S6,S7,S8"
SUBJECTS_TEST = "S9,S11"
SUBJECTS_UNLABELLED = ""

ARCHITECTURE = [3, 3, 3, 3, 3]
DROPOUT = 0.25
CAUSAL = False
CHANNELS = 1024
DENSE = False

CHECKPOINT = "../../VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin"
