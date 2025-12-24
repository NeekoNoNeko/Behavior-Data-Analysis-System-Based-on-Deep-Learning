# pose_cls_lib/skeleton.py
import cv2
import numpy as np
from .config import *

def draw_gray_skeleton(kpts, img_w, img_h, size=CLS_IMG_SIZE):
    img = np.zeros((size, size), dtype=np.uint8)
    pts = []

    for x, y, conf in kpts:
        if conf >= CONF_THR:
            px = int((x / img_w) * size)
            py = int((y / img_h) * size)
            pts.append((px, py))
        else:
            pts.append(None)

    for p in pts:
        if p:
            cv2.circle(img, p, 2, 255, -1)

    for i, j in COCO_SKELETON:
        if pts[i] and pts[j]:
            cv2.line(img, pts[i], pts[j], 255, 1)

    return img
