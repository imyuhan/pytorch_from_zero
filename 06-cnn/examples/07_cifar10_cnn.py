"""05-7 GPU 训练(CIFAR-10 + 简易 CNN)

把所有部件拼起来:DataLoader → 模型 → 训练 → 验证 → GPU
跑 1-2 个 epoch 看效果,完整训练在 06 综合实验。
"""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
os.makedirs(data_root, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 数据(CIFAR-10 下载偶尔会断,加 3 次重试)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # CIFAR-10 3 通道
])

def load_cifar(train, max_retry=3):
    for i in range(max_retry):
        try:
            return datasets.CIFAR10(root=data_root, train=train, download=True, transform=transform)
        except Exception as e:
            print(f"下载失败({i+1}/{max_retry}): {e}")
            if i + 1 == max_retry:
                raise
            import time; time.sleep(5)

train_set = load_cifar(train=True)
test_set = load_cifar(train=False)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

# 简易 CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 32→16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), # 16→8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练 1 epoch
model.train()
t0 = time.time()
for i, (xb, yb) in enumerate(train_loader):
    xb, yb = xb.to(device), yb.to(device)
    loss = criterion(model(xb), yb)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if (i + 1) % 100 == 0:
        print(f"  batch {i+1}/{len(train_loader)}, loss = {loss.item():.4f}")
print(f"训练 1 epoch 用时: {time.time() - t0:.1f}s")

# 验证
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
print(f"测试集准确率: {correct / total * 100:.2f}%")
