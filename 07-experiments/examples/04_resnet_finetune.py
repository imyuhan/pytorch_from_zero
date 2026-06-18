"""06-4 ResNet18 微调 CIFAR-10(迁移学习)

把 04-5 学的"替换分类头 + 冻 backbone"用上,在 CIFAR-10 上看效果。
"""
import argparse
import os
import time
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_loaders(batch_size=128):
    train_tf = transforms.Compose([
        transforms.Resize(224),  # ResNet 期望 224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    val = datasets.CIFAR10(root=data_root, train=False, download=True, transform=val_tf)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val, batch_size=256, shuffle=False, num_workers=0),
    )


def build_model(freeze_backbone=True, num_classes=10):
    model = models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--mode", choices=["head_only", "finetune"], default="head_only")
    args = parser.parse_args()

    freeze = (args.mode == "head_only")
    print(f"模式: {args.mode}({'冻 backbone' if freeze else '全模型微调'})")

    train_loader, val_loader = get_loaders()
    model = build_model(freeze_backbone=freeze).to(device)
    criterion = nn.CrossEntropyLoss()
    # 冻 backbone 时 lr 可以大,全模型微调时 lr 要小
    lr = 1e-3 if freeze else 1e-4
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    best_acc = 0.0
    t0 = time.time()
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
        acc = correct / total
        print(f"epoch {epoch}/{args.epochs}  val_acc = {acc*100:.2f}%")

        # 保存最优权重
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "resnet18_cifar10_best.pt"))
            print(f"  ↳ 新最佳,已保存")

    print(f"\n最佳 val_acc: {best_acc*100:.2f}%,  耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
