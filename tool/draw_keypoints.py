import os
import cv2

def draw_keypoints_on_images(image_folder, keypoint_folder, output_folder):
    """
    从txt文件中读取关键点数据，并将其绘制到对应的图片上。

    Args:
        image_folder (str): 包含原始输入图像的文件夹路径。
        keypoint_folder (str): 包含关键点txt文件的文件夹路径。
        output_folder (str): 保存绘制了关键点图像的文件夹路径。
    """
    print("开始绘制关键点...")
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(keypoint_folder):
        if filename.lower().endswith('.txt'):
            txt_path = os.path.join(keypoint_folder, filename)
            # 根据txt文件名推断原始图片文件名
            # 假设txt文件名是 "image_name.txt" 或 "image_name_X.txt"，原始图片是 "image_name.jpg" 或 "image_name.png"
            # 这里需要更灵活地匹配，因为yolopose.py中保存的txt文件名是基于原始图片名，例如 "000000.txt" 对应 "000000.jpg"
            base_filename = os.path.splitext(filename)[0]
            
            # 尝试匹配常见的图片格式
            image_found = False
            for ext in ('.jpg', '.png', '.jpeg', '.bmp', '.tiff'):
                image_path = os.path.join(image_folder, base_filename + ext)
                if os.path.exists(image_path):
                    image_found = True
                    break
            
            if not image_found:
                print(f"警告: 未找到与 {filename} 对应的图片，跳过。")
                continue

            print(f"正在处理关键点文件: {filename} 及其对应图片: {os.path.basename(image_path)}")

            img = cv2.imread(image_path)
            if img is None:
                print(f"警告: 无法读取图片 {os.path.basename(image_path)}，跳过。")
                continue

            h, w, _ = img.shape # 获取图像的高度和宽度

            # 定义COCO 17关键点的连接对
            POSE_PAIRS = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # 头部：鼻子到眼睛，眼睛到耳朵
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # 胳膊：肩膀到肩膀，肩膀到肘，肘到腕
                (11, 12), (5, 11), (6, 12), # 躯干：髋部到髋部，肩膀到髋部
                (11, 13), (13, 15), (12, 14), (14, 16) # 腿部：髋部到膝，膝到踝
            ]

            # 预定义一些颜色，用于区分不同的人
            COLORS = [
                (0, 255, 0),    # 绿色
                (0, 0, 255),    # 蓝色
                (255, 0, 0),    # 红色
                (0, 255, 255),  # 黄色
                (255, 255, 0),  # 青色
                (255, 0, 255),  # 品红色
                (0, 128, 255),  # 橙色
                (128, 0, 255),  # 紫色
                (0, 255, 128),  # 浅绿色
                (128, 255, 0)   # 柠檬绿
            ]
            color_idx = 0

            all_people_keypoints = [] # 存储所有人的关键点数据
            current_person_keypoints = [None] * 17 # 存储当前人的关键点数据

            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        try:
                            kp_idx = int(parts[0])
                            x = float(parts[1])
                            y = float(parts[2])
                            confidence = float(parts[3])

                            # 如果是新的0号关键点，并且当前人已经有数据，则保存当前人数据并开始新的人
                            if kp_idx == 0 and current_person_keypoints[0] is not None:
                                all_people_keypoints.append(current_person_keypoints)
                                current_person_keypoints = [None] * 17 # 重置为新的人

                            if 0 <= kp_idx < len(current_person_keypoints):
                                current_person_keypoints[kp_idx] = (x, y, confidence)
                        except ValueError:
                            print(f"警告: 无法解析关键点数据行: {line.strip()}，跳过。")
                            continue
            
            # 添加最后一个人的关键点数据
            if current_person_keypoints[0] is not None:
                all_people_keypoints.append(current_person_keypoints)

            # 遍历每个人，绘制关键点和连接线
            for person_keypoints in all_people_keypoints:
                current_color = COLORS[color_idx % len(COLORS)] # 循环使用颜色
                color_idx += 1

                # 在图片上绘制关键点
                for kp_idx, kp_info in enumerate(person_keypoints):
                    if kp_info is not None:
                        x, y, conf = kp_info
                        if conf > 0.5:  # 只绘制置信度较高的关键点
                            center = (int(x * w), int(y * h)) # 将归一化坐标转换为像素坐标
                            radius = 5
                            cv2.circle(img, center, radius, current_color, -1) # 使用当前人的颜色

                # 绘制关键点之间的连接线
                for pair in POSE_PAIRS:
                    partA = pair[0]
                    partB = pair[1]

                    # 确保两个关键点都存在且置信度较高
                    if person_keypoints[partA] is not None and person_keypoints[partB] is not None:
                        x_A, y_A, conf_A = person_keypoints[partA]
                        x_B, y_B, conf_B = person_keypoints[partB]

                        if conf_A > 0.5 and conf_B > 0.5: # 只有当两个关键点都可信时才绘制连接线
                            point_A = (int(x_A * w), int(y_A * h)) # 将归一化坐标转换为像素坐标
                            point_B = (int(x_B * w), int(y_B * h)) # 将归一化坐标转换为像素坐标
                            line_thickness = 2
                            cv2.line(img, point_A, point_B, current_color, line_thickness) # 使用当前人的颜色

            # 保存绘制结果
            output_image_path = os.path.join(output_folder, f"annotated_{os.path.basename(image_path)}")
            cv2.imwrite(output_image_path, img)
            print(f"绘制结果已保存到: {output_image_path}")

    print("所有关键点绘制完成。")

if __name__ == "__main__":
    # 定义输入和输出文件夹
    input_images_dir = r"../data/photo/1"  # 原始图像文件夹
    keypoint_results_dir = "data/keypoint_results" # 关键点txt文件所在的文件夹
    output_annotated_dir = "data/annotated_keypoints" # 绘制结果保存文件夹

    # 运行关键点绘制
    draw_keypoints_on_images(input_images_dir, keypoint_results_dir, output_annotated_dir)