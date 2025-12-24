import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
from torchvision.models import ResNet50_Weights

# 统一参数设置
batch_size = 64           # 每批次样本数
learning_rate = 0.001     # 学习率
num_epochs = 200           # 最大训练轮数
patience = 10              # 早停耐心值

def get_data_loaders(train_dir, val_dir, batch_size=batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_dataset

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    print(f'\nVal set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy:.2f}%)\n')
    return val_loss, accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = r"C:\workspace\resnet\data\train"
    val_dir = r"C:\workspace\resnet\data\val"
    train_loader, val_loader, train_dataset = get_data_loaders(train_dir, val_dir)
    num_classes = len(train_dataset.classes)
    
    # 加载预训练权重
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    # 替换最后的全连接层，适配你的类别数
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    trigger_times = 0

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        val_loss, val_acc = validate(model, device, val_loader, criterion)

        # 早停与模型保存
        if val_acc > best_acc:
            best_acc = val_acc
            trigger_times = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"模型已保存，当前最佳验证准确率: {best_acc:.2f}%")
        else:
            trigger_times += 1
            print(f"验证集准确率未提升，已连续 {trigger_times} 次未提升")
            if trigger_times >= patience:
                print("早停触发，训练终止。")
                break

if __name__ == "__main__":
    main()