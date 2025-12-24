import os
import cv2
from ultralytics import YOLO
import torch # 引入 torch 库以处理 PyTorch 张量数据

def run_keypoint_detection(image_folder, output_folder, model_name='yolo11x-pose.pt'):
    """
    使用 YOLOv8 模型对指定文件夹中的图像进行关键点检测。
    当有多个目标时，保存靠近水平中心且面积较大的对象的关键点。同时可视化。
    关键点结果保存为归一化的坐标。

    Args:
        image_folder (str): 包含输入图像的文件夹路径。
        output_folder (str): 保存带有关键点检测结果的图像的文件夹路径。
        model_name (str): YOLOv8 姿态模型的名称，例如 'yolov8n-pose.pt'。
    """
    # 加载 YOLOv8 姿态模型
    print(f"正在加载 YOLO 姿态模型: {model_name}...")
    model = YOLO(model_name)
    print("模型加载成功。")

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 定义 COCO 关键点骨架连接 (索引从 0 开始)
    # 这是一个常见的 17 点骨架定义，如果您的模型使用不同的关键点顺序，可能需要调整
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4), # 头部和手臂
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # 身体和手臂
        (5, 11), (6, 12), (11, 12), # 躯干
        (11, 13), (13, 15), (12, 14), (14, 16) # 腿部
    ]
    # 关键点颜色 (BGR)
    keypoint_color = (0, 0, 255) # 红色
    skeleton_color = (255, 0, 0) # 蓝色
    line_thickness = 2
    point_radius = 5
    confidence_threshold = 0.5 # 关键点和连接的置信度阈值

    # 遍历输入文件夹中的所有图像
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(image_folder, filename)
            output_image_path = os.path.join(output_folder, f"keypoints_{filename}")

            print(f"正在处理图像: {filename}")

            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                print(f"警告: 无法读取图像 {filename}，跳过。")
                continue

            img_height, img_width, _ = img.shape # 获取图像尺寸，用于归一化

            # 运行关键点检测
            results = model(img, stream=False)

            # 假设每张图片只有一个结果对象 (当 stream=False 时)
            if results:
                current_result = results[0]
                
                best_person_idx = -1
                best_score = -1.0

                img_center_x = img_width / 2

                # 如果检测到目标，则进行筛选
                if current_result.boxes is not None and len(current_result.boxes) > 0:
                    for idx, box in enumerate(current_result.boxes.data):
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box[:4].cpu().numpy() 
                        area = (x2 - x1) * (y2 - y1)
                        person_center_x = (x1 + x2) / 2
                        distance_to_center = abs(person_center_x - img_center_x)

                        # 评分机制：面积越大越好，距离中心越近越好
                        # 1.0 + distance_to_center 避免除以零，并确保距离越小分数越高
                        score = area / (1.0 + distance_to_center) 

                        if score > best_score:
                            best_score = score
                            best_person_idx = idx
                
                # 如果找到了最佳目标
                if best_person_idx != -1:
                    selected_box = current_result.boxes.data[best_person_idx]
                    selected_keypoints = current_result.keypoints.data[best_person_idx]

                    # 可视化：只绘制选定目标的边界框和关键点
                    annotated_frame = img.copy()
                    
                    # 绘制边界框
                    x1, y1, x2, y2 = selected_box[:4].cpu().numpy()
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), line_thickness) # 绿色边界框

                    # 绘制关键点和骨架连接
                    if selected_keypoints is not None and selected_keypoints.numel() > 0:
                        keypoints_coords = {} # 存储关键点坐标，方便绘制骨架
                        for kp_idx, kp in enumerate(selected_keypoints):
                            x, y, conf = kp[0].item(), kp[1].item(), kp[2].item()
                            if conf > confidence_threshold: # 只绘制置信度大于阈值的关键点
                                cv2.circle(annotated_frame, (int(x), int(y)), point_radius, keypoint_color, -1) # 红色圆点
                                keypoints_coords[kp_idx] = (int(x), int(y))
                        
                        # 绘制骨架连接
                        for start_kp_idx, end_kp_idx in skeleton:
                            if start_kp_idx in keypoints_coords and end_kp_idx in keypoints_coords:
                                start_point = keypoints_coords[start_kp_idx]
                                end_point = keypoints_coords[end_kp_idx]
                                cv2.line(annotated_frame, start_point, end_point, skeleton_color, line_thickness) # 蓝色线条

                    cv2.imwrite(output_image_path, annotated_frame)
                    print(f"选定对象的关键点检测结果已保存到: {output_image_path}")

                    # 将选定对象的关键点数据保存到 txt 文件 (归一化坐标)
                    txt_filename = os.path.splitext(filename)[0] + ".txt"
                    txt_output_path = os.path.join(output_folder, txt_filename)

                    with open(txt_output_path, 'w') as f:
                        for kp_idx, kp in enumerate(selected_keypoints):
                            x, y, conf = kp[0].item(), kp[1].item(), kp[2].item()
                            # 归一化坐标
                            normalized_x = x / img_width
                            normalized_y = y / img_height
                            f.write(f"{kp_idx} {normalized_x:.4f} {normalized_y:.4f} {conf:.2f}\n") # 保留四位小数
                    print(f"选定对象的归一化关键点数据已保存到: {txt_output_path}")
                else:
                    print(f"在图像 {filename} 中未找到符合条件的对象。")
            else:
                print(f"在图像 {filename} 中未检测到任何对象。")

if __name__ == "__main__":
    # 定义输入和输出文件夹
    input_images_dir = "test"  # 您的图像文件夹
    output_results_dir = "tmp" # 关键点检测结果保存文件夹

    # 运行关键点检测
    run_keypoint_detection(input_images_dir, output_results_dir)
    print("所有图像的关键点检测已完成。")