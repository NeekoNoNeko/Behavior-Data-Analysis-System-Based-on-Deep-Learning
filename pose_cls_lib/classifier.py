# pose_cls_lib/classifier.py
import cv2
import numpy as np
import onnxruntime as ort
from .config import CLS_IMG_SIZE, DEFAULT_CLASS_NAMES

class SkeletonClassifier:
    def __init__(self, onnx_path, class_names=None):
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.sess = ort.InferenceSession(
            onnx_path, providers=['CPUExecutionProvider']
        )
        self.input_name = self.sess.get_inputs()[0].name

    def infer(self, gray_img):
        img = cv2.resize(gray_img, (CLS_IMG_SIZE, CLS_IMG_SIZE))
        img = np.stack([img, img, img], axis=0)
        img = img.astype(np.float32) / 255.0
        img = img[None]

        logits = self.sess.run(None, {self.input_name: img})[0][0]
        cls_id = int(np.argmax(logits))
        return self.class_names[cls_id]
