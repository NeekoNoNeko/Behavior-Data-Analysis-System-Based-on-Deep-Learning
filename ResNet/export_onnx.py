import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
import os
import onnx
from onnx import external_data_helper

# ------------------------
# 参数
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_model.pth"
train_dir = r"C:\workspace\resnet\data\train"
onnx_path = "resnet50_finetune.onnx"
single_onnx_path = "resnet50_finetune_single.onnx"

# ------------------------
# 类别
# ------------------------
classes = sorted(os.listdir(train_dir))
num_classes = len(classes)

# ------------------------
# 模型
# ------------------------
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

dummy_input = torch.randn(1, 3, 224, 224, device=device)

# ------------------------
# 1. 导出 ONNX（可能生成 .onnx.data）
# ------------------------
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    opset_version=20,
    do_constant_folding=True,
)

# ------------------------
# 2. 合并 external data -> single onnx
# ------------------------
onnx_model = onnx.load(onnx_path)
external_data_helper.convert_model_from_external_data(onnx_model)
onnx.save(onnx_model, single_onnx_path)

# ------------------------
# 3. 清理中间文件
# ------------------------
def safe_remove(path):
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"🗑 已删除: {path}")
        except Exception as e:
            print(f"⚠️ 删除失败 {path}: {e}")

safe_remove(onnx_path)
safe_remove(onnx_path + ".data")

print(f"\n✅ 最终保留文件: {single_onnx_path}")
