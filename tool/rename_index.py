import os
import re

def rename_images_in_folder(folder_path):
    """
    重命名指定文件夹中的所有图片文件为 '0_索引.jpg' 格式。
    图片文件会按照其原始文件名进行排序，然后依次编号。

    Args:
        folder_path (str): 包含图片文件的文件夹路径。
    """
    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在。")
        return

    print(f"正在处理文件夹: {folder_path}")

    image_files = []
    for filename in os.listdir(folder_path):
        # 检查文件扩展名，只处理常见的图片格式
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            image_files.append(filename)

    # 对文件进行排序，以确保重命名顺序一致
    # 这里使用自然排序，例如 '1.jpg', '10.jpg', '2.jpg' 会被正确排序为 '1.jpg', '2.jpg', '10.jpg'
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    image_files.sort(key=natural_sort_key)

    for index, old_filename in enumerate(image_files):
        # 构建新的文件名
        new_filename = f"5_{index}.jpg"
        old_filepath = os.path.join(folder_path, old_filename)
        new_filepath = os.path.join(folder_path, new_filename)

        try:
            os.rename(old_filepath, new_filepath)
            print(f"已将 '{old_filename}' 重命名为 '{new_filename}'")
        except OSError as e:
            print(f"重命名文件 '{old_filename}' 失败: {e}")

    print("所有图片重命名完成。")

if __name__ == "__main__":
    # 定义您的图片文件夹路径
    images_directory = r"..\data\图片分类\驼背"

    # 运行重命名程序
    rename_images_in_folder(images_directory)
