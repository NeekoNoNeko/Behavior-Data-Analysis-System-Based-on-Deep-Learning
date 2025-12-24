import cv2
import numpy as np
import os


def draw_skeleton_from_keypoints(keypoint_file_path, output_image_path, image_width=640, image_height=640, confidence_threshold=0.8):
    """
    根据归一化的人体关键点数据绘制骨架。

    Args:
        keypoint_file_path (str): 包含归一化关键点数据的TXT文件路径。
        output_image_path (str): 绘制骨架后图像的保存路径。
        image_width (int): 输出图像的宽度。
        image_height (int): 输出图像的高度。
        confidence_threshold (float): 绘制关键点的置信度阈值。只有置信度高于此阈值的关键点才会被绘制。
    """
    # 创建一个黑色的背景图像 (灰度图)
    image = np.zeros((image_height, image_width), dtype=np.uint8)

    keypoints = [None] * 17  # 初始化一个列表来存储17个关键点，索引对应关键点ID

    try:
        with open(keypoint_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:  # 期望包含ID, x, y, confidence
                    try:
                        kp_id = int(parts[0])
                        x = float(parts[1]) * image_width
                        y = float(parts[2]) * image_height
                        confidence = float(parts[3])

                        if 0 <= kp_id < 17:  # 确保关键点ID在有效范围内
                            if confidence > confidence_threshold:
                                keypoints[kp_id] = (int(x), int(y))
                            else:
                                print(f"Warning: Keypoint ID {kp_id} skipped due to low confidence ({confidence} <= {confidence_threshold}). Line: {line.strip()}")
                        else:
                            print(f"Warning: Keypoint ID {kp_id} out of expected range (0-16). Line: {line.strip()}")
                    except ValueError:
                        print(f"Warning: Could not parse line: {line.strip()}")
                        continue
                else:
                    print(f"Warning: Incorrect line format (expected 4 parts: ID, x, y, confidence). Skipping line: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Keypoint file not found at {keypoint_file_path}")
        return
    except Exception as e:
        print(f"Error reading keypoint file: {e}")
        return

    # 过滤掉未找到的关键点
    valid_keypoints = [(kp_id, kp_coord) for kp_id, kp_coord in enumerate(keypoints) if kp_coord is not None]
    if not valid_keypoints:
        print("No valid keypoints found in the file after applying confidence threshold. Skipping drawing.")
        return

    # 定义COCO关键点连接（17个关键点）
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

    # 骨架连接定义 (索引对)
    # 这是一个常见的COCO骨架连接定义，您可以根据实际的关键点模型进行调整
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (11, 12), (5, 11), (6, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]

    # 绘制关键点
    for kp_id, kp_coord in enumerate(keypoints):
        if kp_coord is not None:
            cv2.circle(image, kp_coord, 2, 255, -1)  # 白色圆点表示关键点

    # 绘制骨架连接
    for connection in connections:
        p1_idx, p2_idx = connection
        if keypoints[p1_idx] is not None and keypoints[p2_idx] is not None:
            p1 = keypoints[p1_idx]
            p2 = keypoints[p2_idx]
            cv2.line(image, p1, p2, 255, 1)  # 白色线条表示骨架

    # 保存图像
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, image)
    print(f"Skeleton image saved to {output_image_path}")


if __name__ == "__main__":
    # 定义输入关键点文件所在的根目录
    # 假设关键点文件位于此目录下的子文件夹中，例如 data/keypoint/0/, data/keypoint/1/ 等
    input_keypoint_base_dir = r"C:\workspace\Behavior-Data-Analysis-System-Based-on-Deep-Learning\data\keypoint"
    
    # 定义输出骨架图像的根目录
    # 生成的骨架图像将保存在此目录下的对应子文件夹中
    output_base_dir = r"/data/output_skeletons"

    # 遍历 input_keypoint_base_dir 下的所有子目录
    # 例如，如果 input_keypoint_base_dir 是 'data/keypoint'，它会遍历 '0', '1', ... 等子文件夹
    for sub_dir_name in os.listdir(input_keypoint_base_dir):
        input_keypoint_dir = os.path.join(input_keypoint_base_dir, sub_dir_name)

        # 确保当前项是一个目录
        if os.path.isdir(input_keypoint_dir):
            # 为当前的子目录创建对应的输出目录
            output_image_sub_dir = os.path.join(output_base_dir, sub_dir_name)
            os.makedirs(output_image_sub_dir, exist_ok=True) # 如果目录不存在则创建

            print(f"正在处理目录中的关键点文件: {input_keypoint_dir}")
            print(f"骨架图像将保存到: {output_image_sub_dir}")

            # 遍历当前子目录下的所有文件
            for filename in os.listdir(input_keypoint_dir):
                # 只处理以 .txt 结尾的关键点文件
                if filename.endswith(".txt"):
                    keypoint_file = os.path.join(input_keypoint_dir, filename)
                    
                    # 从关键点文件名中提取不带扩展名的部分，用于构建输出图像的文件名
                    # 例如，对于 "0_0.txt"，file_name_without_ext 将是 "0_0"
                    file_name_without_ext = os.path.splitext(os.path.basename(keypoint_file))[0]
                    
                    # 构建输出图像的完整路径和文件名
                    # 例如，对于 "0_0.txt"，输出图像将是 "keypoints_0_0.jpg"
                    output_image_file = os.path.join(output_image_sub_dir, f"keypoints_{file_name_without_ext}.jpg")

                    print(f"正在为 {keypoint_file} 绘制骨架...")
                    # 调用函数绘制骨架，并指定图像尺寸
                    draw_skeleton_from_keypoints(keypoint_file, output_image_file, image_width=224, image_height=224)
                    print(f"已完成 {keypoint_file} 的骨架绘制。")

    print("所有关键点文件处理完毕。")

    # 您也可以尝试显示图像 (需要安装matplotlib或直接使用cv2.imshow)
    # cv2.imshow("Skeleton", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()