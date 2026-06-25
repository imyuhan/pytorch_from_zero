"""05-6 模型保存与加载

只保存参数(推荐,跨 PyTorch 版本兼容),不保存整个模型。
"""
import os
import torch
import torch.nn as nn

# 路径
ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, "demo_model.pt")

# 假模型
model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

# --- 保存 ---
# 推荐:只保存 state_dict(参数字典)
torch.save(model.state_dict(), ckpt_path)
print(f"已保存到: {ckpt_path}")
print(f"文件大小: {os.path.getsize(ckpt_path)} 字节")

# --- 加载 ---
new_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
new_model.load_state_dict(torch.load(ckpt_path, weights_only=True))  # weights_only 防 pickle 漏洞
new_model.eval()
print(f"已加载,权重匹配: ", end="")
x = torch.randn(1, 10)
print(torch.allclose(model(x), new_model(x)))

# --- 顺手保存 optimizer 和 epoch(训练断点续训用)---
# torch.save({
#     "epoch": 5,
#     "model_state_dict": model.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
#     "loss": 0.123,
# }, "checkpoint.pth")
