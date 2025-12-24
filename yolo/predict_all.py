# -*- coding: utf-8 -*-
"""
摄像头实时姿态检测 + GUI 展示
Author : your_name
"""
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk   # pip install pillow

# ========= 需要改的两个权重路径 =========
POSE_MODEL = 'yolo11n-pose.pt'          # YOLO-Pose
CLS_MODEL  = 'runs/cls/skeletons/weights/best.pt'  # YOLO-CLS
# =======================================

# 全局变量
COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),        # head
    (5,6),(5,7),(7,9),(6,8),(8,10), # arms
    (11,12),(5,11),(6,12),          # torso
    (11,13),(13,15),(12,14),(14,16) # legs
]

CLS_NAMES = None

# ---------- 1. 加载模型 ----------
pose_model = YOLO(POSE_MODEL, task='pose')
cls_model  = YOLO(CLS_MODEL,  task='classify')
CLS_NAMES  = cls_model.names

# ---------- 2. 画 224 灰度骨架 ----------
def draw_gray_skeleton(kpts, img_w, img_h, conf_thr=0.3, size=224):
    img = np.zeros((size, size), dtype=np.uint8)
    kpts_np = kpts.cpu().numpy()
    pts = []
    for x, y, conf in kpts_np:
        if conf >= conf_thr:
            x_norm = x / img_w
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

# ---------- 3. 单帧推理 ----------
def inference_one_frame(frame_bgr):
    """
    frame_bgr: 摄像头原始 BGR 图
    return:
        vis:      带骨架/框的 BGR 图
        skel_gray:224×224 灰度骨架图
        label:    分类字符串
    """
    h, w = frame_bgr.shape[:2]
    results = pose_model(frame_bgr, stream=False)[0]

    # 取最“居中+最大”的人
    if results.keypoints is None or len(results.keypoints.data) == 0:
        # 没检测到人，返回原图 + 空白骨架 + 标签“None”
        vis = frame_bgr.copy()
        skel_gray = np.zeros((224, 224), dtype=np.uint8)
        return vis, skel_gray, "None"

    img_cx = w / 2
    best_idx, best_score = -1, -1
    for idx, box in enumerate(results.boxes.data):
        x1, y1, x2, y2 = box[:4].cpu().numpy()
        area = (x2 - x1) * (y2 - y1)
        cx = (x1 + x2) / 2
        score = area / (1.0 + abs(cx - img_cx))
        if score > best_score:
            best_score, best_idx = score, idx

    if best_idx == -1:
        vis = frame_bgr.copy()
        skel_gray = np.zeros((224, 224), dtype=np.uint8)
        return vis, skel_gray, "None"

    # 可视化
    vis = results.plot()

    # 灰度骨架
    kpts = results.keypoints.data[best_idx]
    skel_gray = draw_gray_skeleton(kpts, w, h)

    # 分类
    top1_id = int(cls_model(skel_gray, stream=False)[0].probs.top1)
    label = CLS_NAMES[top1_id]
    return vis, skel_gray, label

# ---------- 4. GUI 部分 ----------
class App:
    def __init__(self, window, window_title):
        self.window = window
        window.title(window_title)
        self.vid = cv2.VideoCapture(0)   # 0 为默认摄像头
        self.delay = 15                  # ms

        # 左侧大标签显示摄像头
        self.main_panel = Label(window)
        self.main_panel.grid(row=0, column=0, padx=10, pady=10)

        # 右上角小标签显示灰度骨架
        self.skel_panel = Label(window)
        self.skel_panel.grid(row=0, column=1, padx=10, pady=10)

        # 右下角文字标签
        self.label_text = tk.StringVar()
        self.label_text.set("分类结果: None")
        self.label_widget = Label(window, textvariable=self.label_text,
                                  font=("Arial", 16))
        self.label_widget.grid(row=1, column=1, padx=10, pady=10)

        self.update()
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.read()
        if not ret:
            self.on_close()
            return

        vis, skel_gray, label = inference_one_frame(frame)

        # 把 BGR 转 RGB，再转 ImageTk
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(vis_rgb)
        imgtk = ImageTk.PhotoImage(image=im)
        self.main_panel.imgtk = imgtk
        self.main_panel.config(image=imgtk)

        # 灰度骨架同样处理
        skel_rgb = cv2.cvtColor(skel_gray, cv2.COLOR_GRAY2RGB)
        skel_im = Image.fromarray(skel_rgb)
        skel_imgtk = ImageTk.PhotoImage(image=skel_im)
        self.skel_panel.imgtk = skel_imgtk
        self.skel_panel.config(image=skel_imgtk)

        # 更新文字
        self.label_text.set(f"分类结果: {label}")

        self.window.after(self.delay, self.update)

    def on_close(self):
        self.vid.release()
        self.window.destroy()

# ---------- 5. 启动 ----------
if __name__ == '__main__':
    root = tk.Tk()
    App(root, "实时姿态检测 Demo")