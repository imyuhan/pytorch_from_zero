"""06-3 数据增强对 CIFAR-10 准确率的影响

对比两组超参:
  A. 不做增强
  B. 加 RandomCrop + RandomHorizontalFlip

观察 val_acc 差距。
"""
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
device = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 32→16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), # 16→8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_loaders(use_aug):
    if use_aug:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    val = datasets.CIFAR10(root=data_root, train=False, download=True, transform=val_tf)
    return (
        DataLoader(train, batch_size=128, shuffle=True, num_workers=0),
        DataLoader(val, batch_size=256, shuffle=False, num_workers=0),
    )


def train(args, use_aug):
    train_loader, val_loader = get_loaders(use_aug)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(1)
                correct += (preds == yb).sum().item(); total += xb.size(0)
        print(f"  [aug={use_aug}] epoch {epoch}: val_acc = {correct / total * 100:.2f}%")
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    print("=" * 50)
    print("A. 不做数据增强")
    print("=" * 50)
    t0 = time.time(); acc_a = train(args, use_aug=False); ta = time.time() - t0

    print("\n" + "=" * 50)
    print("B. 加 RandomCrop + Flip")
    print("=" * 50)
    t0 = time.time(); acc_b = train(args, use_aug=True); tb = time.time() - t0

    print(f"\n结果: A={acc_a*100:.2f}% ({ta:.0f}s)  B={acc_b*100:.2f}% ({tb:.0f}s)")
    print(f"提升: {(acc_b - acc_a) * 100:+.2f}%")


if __name__ == "__main__":
    main()
