import os
import shutil
import random

# 原始数据路径
base_dir = "../data/keypoint"
output_dir = "../data/gesture_cls2"
train_ratio = 0.8

# 获取所有类别
categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
categories.sort()

# 创建目标目录结构
for subset in ["train", "val"]:
    for category in categories:
        os.makedirs(os.path.join(output_dir, subset, category), exist_ok=True)

for category in categories:
    img_dir = os.path.join(base_dir, category)
    images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for img_name in train_imgs:
        src_img_path = os.path.join(img_dir, img_name)
        dst_img_path = os.path.join(output_dir, "train", category, img_name)
        shutil.copy(src_img_path, dst_img_path)

    for img_name in val_imgs:
        src_img_path = os.path.join(img_dir, img_name)
        dst_img_path = os.path.join(output_dir, "val", category, img_name)
        shutil.copy(src_img_path, dst_img_path)

print("图片已按类别和比例划分到 data/gesture_cls/train 和 data/gesture_cls/val 目录下，结构适用于分类任务。")