# pose_cls_lib/infer.py
import cv2
from .pose import PoseONNX
from .skeleton import draw_gray_skeleton
from .classifier import SkeletonClassifier
from .config import POSE_ONNX, CLS_ONNX

class PoseClsInferencer:
    def __init__(self, pose_onnx, cls_onnx, class_names):
        self.pose = PoseONNX(pose_onnx)
        self.cls  = SkeletonClassifier(cls_onnx, class_names)

    def infer(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                raise ValueError("无法读取图片")

        kpts = self.pose.infer(img)
        if kpts is None:
            return "no_person"

        skel = draw_gray_skeleton(kpts, img.shape[1], img.shape[0])
        return self.cls.infer(skel)


def create_inferencer(
    pose_onnx=POSE_ONNX,
    cls_onnx=CLS_ONNX,
    class_names=None
):
    inferencer = PoseClsInferencer(
        pose_onnx=pose_onnx,
        cls_onnx=cls_onnx,
        class_names=class_names
    )
    return inferencer.infer
