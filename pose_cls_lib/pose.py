# pose_cls_lib/pose.py
import cv2
import numpy as np
import onnxruntime as ort
from .config import POSE_IMG_SIZE, CONF_THR

class PoseONNX:
    def __init__(self, onnx_path):
        self.sess = ort.InferenceSession(
            onnx_path, providers=['CPUExecutionProvider']
        )
        self.input_name = self.sess.get_inputs()[0].name

    def letterbox(self, img):
        h, w = img.shape[:2]
        scale = POSE_IMG_SIZE / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh))
        canvas = np.zeros((POSE_IMG_SIZE, POSE_IMG_SIZE, 3), dtype=np.uint8)
        canvas[:nh, :nw] = resized
        return canvas, scale

    def infer(self, img_bgr):
        img, scale = self.letterbox(img_bgr)
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None]

        output = self.sess.run(None, {self.input_name: img})[0]
        pred = output[0].transpose(1, 0)

        img_h, img_w = img_bgr.shape[:2]
        best_idx, best_score = -1, -1

        for i, det in enumerate(pred):
            if det[4] < CONF_THR:
                continue
            cx, cy, w, h = det[:4]
            x1 = (cx - w / 2) / scale
            x2 = (cx + w / 2) / scale
            area = w * h
            score = area / (1.0 + abs(x1 + x2 - img_w) / 2)
            if score > best_score:
                best_score, best_idx = score, i

        if best_idx == -1:
            return None

        det = pred[best_idx]
        kpts = det[5:].reshape(17, 3)
        kpts[:, :2] /= scale
        return kpts
