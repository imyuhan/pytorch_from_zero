"""06-5 训练曲线可视化 + 混淆矩阵(纯 matplotlib,不依赖 sklearn/seaborn)

训练时把每 epoch 的指标写到一个 json 文件,最后这个脚本画图。
"""
import os
import matplotlib.pyplot as plt
import numpy as np

# 假数据(实际训练时把每 epoch 的指标记下来再喂进来)
history = {
    "train_loss": [0.45, 0.21, 0.13, 0.09, 0.06],
    "val_loss":   [0.30, 0.25, 0.22, 0.24, 0.27],
    "val_acc":    [0.91, 0.93, 0.94, 0.94, 0.93],
}
out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)

# --- 1. 训练曲线 ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val")
axes[0].set_title("Loss")
axes[0].set_xlabel("epoch")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history["val_acc"], marker="o")
axes[1].set_title("Validation Accuracy")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("acc")
axes[1].grid(True)

out1 = os.path.join(out_dir, "training_curves.png")
plt.savefig(out1, dpi=100, bbox_inches="tight")
print(f"训练曲线已保存: {out1}")
plt.close()

# --- 2. 混淆矩阵(纯 matplotlib 实现 imshow)---
classes = [f"class_{i}" for i in range(10)]
n = 200
rng = np.random.default_rng(42)
y_true = rng.integers(0, 10, n)
y_pred = (y_true + rng.integers(0, 2, n)) % 10  # 大半对,小半错

# 算 10x10 矩阵
cm = np.zeros((10, 10), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10)); ax.set_xticklabels(classes, rotation=45)
ax.set_yticks(range(10)); ax.set_yticklabels(classes)
ax.set_title("Confusion Matrix")
ax.set_ylabel("True")
ax.set_xlabel("Predicted")
# 在格子里写数字
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.colorbar(im)
out2 = os.path.join(out_dir, "confusion_matrix.png")
plt.savefig(out2, dpi=100, bbox_inches="tight")
print(f"混淆矩阵已保存: {out2}")
plt.close()
