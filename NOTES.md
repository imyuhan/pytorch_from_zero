# PyTorch 学习笔记

## 00-setup / 环境验证

- 显卡: NVIDIA GeForce RTX 4060 笔记本版, 8G 显存
- 驱动: 591.74, 支持 CUDA 最高 13.1
- PyTorch wheel: cu124(自带的 CUDA 12.4 runtime,够用)
- `torch.cuda.is_available()` → True
- `torch.cuda.get_device_name(0)` → NVIDIA GeForce RTX 4060 Laptop GPU
