#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双击运行版：把同目录下的 input.mp4 按每秒 2 帧拆成图片，保存到 frames 文件夹
"""
import os
import cv2
from pathlib import Path

VIDEO_FILE = '冯铟壕.mp4'   # 默认视频文件
OUT_DIR    = 'frames'      # 输出目录
TARGET_FPS = 1.0           # 每秒保存几帧

def extract_frames(video_path: str, out_dir: str, fps: float = 2.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'打不开视频：{video_path}')

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps == 0:
        raise RuntimeError('无法获取视频帧率')

    interval = max(1, int(round(src_fps / fps)))
    Path(out_dir).mkdir(exist_ok=True)

    count, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            cv2.imwrite(os.path.join(out_dir, f'{saved:06d}.jpg'), frame)
            saved += 1
        count += 1

    cap.release()
    print(f'完成！共保存 {saved} 张图片到“{out_dir}”文件夹')

if __name__ == '__main__':
    # 如果找不到 input.mp4，给出简单提示
    if not os.path.isfile(VIDEO_FILE):
        input(
            '请先把要处理的视频重命名为 input.mp4，\n'
            '然后放到本脚本同一目录下，再双击运行。\n按回车退出…'
        )
        raise SystemExit()

    extract_frames(VIDEO_FILE, OUT_DIR, TARGET_FPS)