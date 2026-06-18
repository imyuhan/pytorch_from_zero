"""05-5 验证循环 + 准确率

训练完一个 epoch 后,看模型在测试集上的表现。
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 测试集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
test_set = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

# 模型(简化版)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 10),
).to(device)
criterion = nn.CrossEntropyLoss()

# --- 验证(没训练过,准确率约 10% 随机)---
model.eval()  # BN/Dropout 切换;这里模型里没这些,但写上养成习惯
correct = 0
total = 0
total_loss = 0

with torch.no_grad():        # 推理时关闭梯度,省显存
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)

        # 预测类别 = logits argmax
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)

acc = correct / total
avg_loss = total_loss / total
print(f"测试集: {total} 条, 准确率 = {acc*100:.2f}%, 平均 loss = {avg_loss:.4f}")
