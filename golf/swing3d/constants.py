# KEYPOINT DETECTION
MODEL_CONFIG_PATH = (
    "../../../detectron2/configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"
)
MODEL_WEIGHTS_PATH = "../../../model_final_5ad38f.pkl"

SMALL_MODEL_CONFIG_PATH = (
    "../../../detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
)
SMALL_MODEL_WEIGHTS_PATH = "../../../model_final_a6e10b.pkl"

DEBUG_IMAGE_PATH = "../../data/debug_frame.png"
DEBUG_OUTPUT_PATH = "../../data/debug_output.png"

KEYPOINT_MAPPING = {
    0: "crotch",
    1: "right_hip",
    2: "right_knee",
    3: "right foot",
    4: "left hip",
    5: "left knee",
    6: "left foot",
    7: "chest",
    8: "neck",
    9: "nose",
    10: "head",
    11: "left_shoulder",
    12: "left_elbow",
    13: "left_hand",
    14: "right_shoulder",
    15: "right_elbow",
    16: "right_hand",
}

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

JOINTS_LEFT = ([4, 5, 6, 11, 12, 13],)
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
CAUSAL_SHIFT = 0

CHECKPOINT = "../../../VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin"

# MISC
DEBUG_BANNER = (
    """
DDDDDDDDDDDDD      EEEEEEEEEEEEEEEEEEEEEEBBBBBBBBBBBBBBBBB   UUUUUUUU     U
UUUUUUU       GGGGGGGGGGGGG;
D::::::::::::DDD   E::::::::::::::::::::EB::::::::::::::::B  U::::::U     U
::::::U    GGG::::::::::::G;
D:::::::::::::::DD E::::::::::::::::::::EB::::::BBBBBB:::::B U::::::U     U
::::::U  GG:::::::::::::::G;
DDD:::::DDDDD:::::DEE::::::EEEEEEEEE::::EBB:::::B     B:::::BUU:::::U     U
:::::UU G:::::GGGGGGGG::::G;
  D:::::D    D:::::D E:::::E       EEEEEE  B::::B     B:::::B U:::::U     U
:::::U G:::::G       GGGGGG;
  D:::::D     D:::::DE:::::E               B::::B     B:::::B U:::::D     D
:::::UG:::::G              ;
  D:::::D     D:::::DE::::::EEEEEEEEEE     B::::BBBBBB:::::B  U:::::D     D
:::::UG:::::G              ;
  D:::::D     D:::::DE:::::::::::::::E     B:::::::::::::BB   U:::::D     D
:::::UG:::::G    GGGGGGGGGG;
  D:::::D     D:::::DE:::::::::::::::E     B::::BBBBBB:::::B  U:::::D     D
:::::UG:::::G    G::::::::G;
  D:::::D     D:::::DE::::::EEEEEEEEEE     B::::B     B:::::B U:::::D     D
:::::UG:::::G    GGGGG::::G;
  D:::::D     D:::::DE:::::E               B::::B     B:::::B U:::::D     D
:::::UG:::::G        G::::G;
  D:::::D    D:::::D E:::::E       EEEEEE  B::::B     B:::::B U::::::U   U:
:::::U G:::::G       G::::G;
DDD:::::DDDDD:::::DEE::::::EEEEEEEE:::::EBB:::::BBBBBB::::::B U:::::::UUU::
:::::U  G:::::GGGGGGGG::::G;
D:::::::::::::::DD E::::::::::::::::::::EB:::::::::::::::::B   UU::::::::::
:::UU    GG:::::::::::::::G;
D::::::::::::DDD   E::::::::::::::::::::EB::::::::::::::::B      UU::::::::
:UU        GGG::::::GGG:::G;
DDDDDDDDDDDDD      EEEEEEEEEEEEEEEEEEEEEEBBBBBBBBBBBBBBBBB         UUUUUUUU
U             GGGGGG   GGGG;
""".replace(
        "\n", ""
    )
    .replace(";", "\n")
    .split("\n")
)
