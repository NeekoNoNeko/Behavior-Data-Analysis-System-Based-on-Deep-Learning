# -*- coding: utf-8 -*-
"""
一键流程：图片 → 关键点 → 灰度骨架（归一化分母=原图W/H） → 分类字符串
并在 tmp 中保存 YOLO-Pose 检测可视化结果（pose_vis_*.jpg）
"""
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ========= 1. 用户需要修改的 4 个路径 =========
POSE_MODEL = 'yolo11n-pose.pt'          # YOLO-Pose 权重
CLS_MODEL  = r'runs/cls/skeletons/weights/best.pt'  # YOLO-CLS 权重
INPUT_DIR  = 'test'          # 待处理图片文件夹
TMP_DIR    = './tmp'         # 临时保存目录
# ============================================

os.makedirs(TMP_DIR, exist_ok=True)

# ---------- 1. YOLO-Pose 检测 ----------
def detect_keypoints(img_bgr):
    """
    返回:
        kpts: 单个最佳人体的关键点 tensor，形状 (17,3)
        vis:  YOLO-Pose 画好的 BGR 可视化图（含框+关键点+骨架）
    """
    model = YOLO(POSE_MODEL, task='pose')
    results = model(img_bgr, stream=False)[0]   # 单张图
    if results.keypoints is None or len(results.keypoints.data) == 0:
        return None, None

    img_w = img_bgr.shape[1]
    img_cx = img_w / 2
    best_idx, best_score = -1, -1
    for idx, box in enumerate(results.boxes.data):
        x1, y1, x2, y2 = box[:4].cpu().numpy()
        area = (x2 - x1) * (y2 - y1)
        cx = (x1 + x2) / 2
        score = area / (1.0 + abs(cx - img_cx))
        if score > best_score:
            best_score, best_idx = score, idx

    if best_idx == -1:
        return None, None

    vis = results.plot()          # BGR numpy 数组
    return results.keypoints.data[best_idx], vis


# ---------- 2. 画 224×224 灰度骨架 ----------
COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),        # head
    (5,6),(5,7),(7,9),(6,8),(8,10), # arms
    (11,12),(5,11),(6,12),          # torso
    (11,13),(13,15),(12,14),(14,16) # legs
]

def draw_gray_skeleton(kpts, img_w, img_h, conf_thr=0.3, size=224):
    """
    kpts: tensor (17,3) → x,y,conf  绝对像素坐标
    img_w, img_h: 原图宽高，用于归一化
    返回 uint8 灰度图 (H,W)
    """
    img = np.zeros((size, size), dtype=np.uint8)
    kpts_np = kpts.cpu().numpy()

    pts = []
    for x, y, conf in kpts_np:
        if conf >= conf_thr:
            x_norm = x / img_w   # 用原图W/H归一化
            y_norm = y / img_h
            pts.append((int(x_norm * size), int(y_norm * size)))
        else:
            pts.append(None)

    # 画点
    for p in pts:
        if p:
            cv2.circle(img, p, 2, 255, -1)
    # 画线
    for i, j in COCO_SKELETON:
        if pts[i] and pts[j]:
            cv2.line(img, pts[i], pts[j], 255, 1)
    return img


# ---------- 3. YOLO-CLS 分类 ----------
CLS_NAMES = None

def classify_skeleton(gray_img):
    global CLS_NAMES
    model = YOLO(CLS_MODEL, task='classify')
    if CLS_NAMES is None:
        CLS_NAMES = model.names
    results = model(gray_img, stream=False)[0]
    top1_id = int(results.probs.top1)
    return CLS_NAMES[top1_id]


# ---------- 主流程 ----------
def main():
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
            continue
        img_path = os.path.join(INPUT_DIR, fname)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print('跳过无法读取的图像:', fname)
            continue

        # 1. 检测关键点 + 获取可视化图
        kpts, vis = detect_keypoints(img_bgr)
        if kpts is None:
            print('未检测到人体:', fname)
            continue

        # 保存 YOLO-Pose 可视化结果
        pose_vis_file = os.path.join(TMP_DIR, f'pose_vis_{fname}')
        cv2.imwrite(pose_vis_file, vis)

        # 2. 画骨架（归一化分母=原图W/H）
        skeleton = draw_gray_skeleton(kpts, img_bgr.shape[1], img_bgr.shape[0])
        skel_file = os.path.join(TMP_DIR, f'skel_{fname}')
        cv2.imwrite(skel_file, skeleton)

        # 3. 分类
        label = classify_skeleton(skeleton)
        print(f'{fname}  →  分类结果: {label}')


if __name__ == '__main__':
    main()