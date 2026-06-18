# 00 - 环境验证 (Environment Check)

> **难度**: ⭐
> **前置**: 无(项目入口第一步)
> **预计耗时**: 5 分钟
> **硬件**: GPU 推荐(CPU 也能跑,只验证不训练)

验证 PyTorch + CUDA 是否可用。

> 📖 详细讲解看 [`doc/00-setup.md`](doc/00-setup.md)

运行:
```powershell
.venv\Scripts\python.exe 00-setup\check_env.py
```

预期输出:
- torch 版本
- CUDA 是否可用
- GPU 名称
- 一个 5x3 随机 tensor(在 GPU 上算的)
