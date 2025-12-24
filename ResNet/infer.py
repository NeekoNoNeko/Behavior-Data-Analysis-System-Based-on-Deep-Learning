import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
import os

# 参数设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_model.pth"
train_dir = r"C:\workspace\resnet\data\train"  # 用于获取类别标签
img_path = "test.jpg"  # 可以是单张图片路径，也可以是文件夹路径

# 加载类别标签
classes = sorted(os.listdir(train_dir))

# 图像预处理（与训练一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 构建模型并加载权重
num_classes = len(classes)
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

def predict(img_file):
    img = Image.open(img_file).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()
        return classes[pred]

if os.path.isfile(img_path):
    # 单张图片推理
    result = predict(img_path)
    print(f"{img_path} -> 预测类别: {result}")
elif os.path.isdir(img_path):
    # 文件夹批量推理
    for fname in os.listdir(img_path):
        fpath = os.path.join(img_path, fname)
        if os.path.isfile(fpath) and fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            result = predict(fpath)
            print(f"{fname} -> 预测类别: {result}")
else:
    print("输入路径无效，请检查 img_path 设置。")