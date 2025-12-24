# pose_cls_lib/config.py

POSE_ONNX = "pose.onnx"
CLS_ONNX  = "cls.onnx"

POSE_IMG_SIZE = 640
CLS_IMG_SIZE  = 224
CONF_THR = 0.3

DEFAULT_CLASS_NAMES = [
    "0other",
    "1sit",
    "2resting_chin",
    "3hunchback"
]

COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (11,12),(5,11),(6,12),
    (11,13),(13,15),(12,14),(14,16)
]
