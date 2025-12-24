# -*- coding: utf-8 -*-
"""
ONNX 一键流程：
图片 → Pose ONNX → 关键点 → 灰度骨架 → CLS ONNX → 分类
"""

import os
import cv2
import numpy as np
import onnxruntime as ort

# ========= 路径 =========
POSE_ONNX = 'pose.onnx'
CLS_ONNX  = r'C:\workspace\Behavior-Data-Analysis-System-Based-on-Deep-Learning\ResNet\resnet50_finetune_single.onnx'
INPUT_DIR = 'test'
TMP_DIR   = './tmp'
os.makedirs(TMP_DIR, exist_ok=True)

# ========= 基本参数 =========
POSE_IMG_SIZE = 640
CLS_IMG_SIZE  = 224
CONF_THR = 0.3

# ========= COCO 骨架 =========
COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (11,12),(5,11),(6,12),
    (11,13),(13,15),(12,14),(14,16)
]

# ========= ONNX Session =========
pose_sess = ort.InferenceSession(POSE_ONNX, providers=['CPUExecutionProvider'])
cls_sess  = ort.InferenceSession(CLS_ONNX,  providers=['CPUExecutionProvider'])

pose_input_name = pose_sess.get_inputs()[0].name
cls_input_name  = cls_sess.get_inputs()[0].name

# ---------- 1. 预处理 ----------
def letterbox(img, new_size=640):
    h, w = img.shape[:2]
    scale = new_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((new_size, new_size, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, scale

def preprocess_pose(img_bgr):
    img, scale = letterbox(img_bgr, POSE_IMG_SIZE)
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]
    return img, scale

# ---------- 2. Pose ONNX 推理 ----------
def detect_keypoints_onnx(img_bgr):
    inp, scale = preprocess_pose(img_bgr)
    output = pose_sess.run(None, {pose_input_name: inp})[0]

    pred = output[0].transpose(1, 0)  # (8400, 56)

    img_h, img_w = img_bgr.shape[:2]
    img_cx = img_w / 2

    best_idx, best_score = -1, -1

    for i, det in enumerate(pred):
        obj = det[4]
        if obj < CONF_THR:
            continue

        cx, cy, w, h = det[:4]
        x1 = (cx - w / 2) / scale
        x2 = (cx + w / 2) / scale

        area = w * h
        score = area / (1.0 + abs(x1 + x2 - img_w) / 2)

        if score > best_score:
            best_score, best_idx = score, i

    if best_idx == -1:
        return None, None

    det = pred[best_idx]

    # -------- 关键点解析 --------
    kpts = det[5:].reshape(17, 3)
    kpts[:, :2] /= scale

    # -------- 可视化 --------
    vis = img_bgr.copy()
    for x, y, c in kpts:
        if c > CONF_THR:
            cv2.circle(vis, (int(x), int(y)), 3, (0,255,0), -1)

    for i, j in COCO_SKELETON:
        if kpts[i,2] > CONF_THR and kpts[j,2] > CONF_THR:
            cv2.line(
                vis,
                (int(kpts[i,0]), int(kpts[i,1])),
                (int(kpts[j,0]), int(kpts[j,1])),
                (255,0,0), 2
            )

    return kpts, vis


# ---------- 3. 画灰度骨架 ----------
def draw_gray_skeleton(kpts, img_w, img_h, size=224):
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

# ---------- 4. CLS ONNX ----------
CLS_NAMES = ['0other', '1sit', '2resting_chin', '3hunchback']  # ⚠️换成你真实类别

def classify_skeleton_onnx(gray_img):
    img = cv2.resize(gray_img, (CLS_IMG_SIZE, CLS_IMG_SIZE))

    # 灰度 → 3 通道
    img = np.stack([img, img, img], axis=0)  # (3,H,W)

    img = img.astype(np.float32) / 255.0
    img = img[None]  # (1,3,H,W)

    logits = cls_sess.run(None, {cls_input_name: img})[0][0]
    cls_id = int(np.argmax(logits))
    return CLS_NAMES[cls_id]


# ---------- 主流程 ----------
def main():
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(('.jpg','.png','.jpeg','.bmp')):
            continue

        img_path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        kpts, vis = detect_keypoints_onnx(img)
        if kpts is None:
            print('未检测到人体:', fname)
            continue

        cv2.imwrite(os.path.join(TMP_DIR, f'pose_vis_{fname}'), vis)

        skeleton = draw_gray_skeleton(kpts, img.shape[1], img.shape[0])
        cv2.imwrite(os.path.join(TMP_DIR, f'skel_{fname}'), skeleton)

        label = classify_skeleton_onnx(skeleton)
        print(f'{fname} → 分类结果: {label}')

if __name__ == '__main__':
    main()
