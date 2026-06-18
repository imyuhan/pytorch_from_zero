# 00 - 环境验证(教学文档)

> 本节唯一目标:**确认 PyTorch 装好、GPU 能用、tensor 能在 GPU 上跑**。一切深度学习任务都建立在这个基础上,环境不对,后面全白搭。

## 0.1 涉及的文件

| 文件 | 作用 |
|------|------|
| `check_env.py` | 自检脚本,打印 torch / CUDA / GPU 关键信息,跑一个 GPU 上的小 tensor |

```powershell
.venv\Scripts\python.exe 00-setup\check_env.py
```

## 0.2 基础知识

### 0.2.1 什么是 PyTorch

PyTorch 是 Meta(原 Facebook)开源的**深度学习框架**。它的核心卖点:

- **动态计算图**(define-by-run):写出来的 Python 代码怎么跑,计算图就怎么长,debug 友好
- **GPU 加速**:一行 `.to("cuda")` 就能把运算搬到 GPU
- **自动求导**:`requires_grad=True` + `.backward()` 自动算梯度,不用手写链式法则
- **生态丰富**:torchvision(图像)、torchaudio(声音)、torchtext(文本)三个官方扩展

### 0.2.2 关键名词

| 名词 | 一句话解释 |
|------|-----------|
| **Tensor** | 多维数组,框架里所有数据的"原子单位" |
| **CUDA** | NVIDIA 给 GPU 编程的并行计算平台;PyTorch 通过它把 tensor 算在 GPU 上 |
| **cuDNN** | NVIDIA 专门为深度学习算子(卷积、池化、归一化等)优化过的 GPU 库,PyTorch 装了 CUDA 之后默认会用 |
| **driver / runtime** | 显卡**驱动**是连接操作系统和显卡的桥梁;CUDA **runtime** 是 PyTorch 自带的,跟驱动版本匹配就能跑 |
| **显存** | GPU 上的内存,模型参数 + 激活值 + 优化器状态都放这里 |

### 0.2.3 驱动、CUDA Toolkit、PyTorch 三者关系

很多人装 PyTorch 时栽在这 —— 简单版:

| 概念 | 你**需要装**吗? |
|------|---------------|
| 显卡驱动(GeForce / Studio) | ✅ 必须,装最新就行 |
| CUDA Toolkit(nvcc / cudnn-dev) | ❌ **不需要**,PyTorch 自带 runtime |
| PyTorch(从官网装 wheel) | ✅ 必须,选 `cu124` 后缀就自带 CUDA 12.4 runtime |

**判断口诀**:`nvidia-smi` 看到的 `CUDA Version: 13.1` 是驱动能支持的**最高版本**,你装的 PyTorch 必须是 ≤ 这个数字。**驱动是上限,PyTorch 自带 runtime 是具体跑的版本**。

比如本机:
- 驱动支持最高 CUDA 13.1
- 装的 PyTorch 2.6.0+cu124 → 自带 runtime 是 CUDA 12.4
- 12.4 ≤ 13.1,**完美兼容**

### 0.2.4 `.venv` 是什么

`venv` 是 Python 3.3+ 自带的**虚拟环境**机制,作用:

- 隔离依赖:A 项目的 PyTorch 不会污染 B 项目的 NumPy
- 不污染系统 Python:不装到 `C:\Program Files\python3.14\Lib\site-packages` 这种系统位置
- 跨机器迁移:只要 Python 版本和 `requirements.txt` 一致,任何机器能复现

`pytorch-from-zero/.venv` 是用 **uv**(一个比 pip 快 10x 的工具)创建的虚拟环境,里面已经预装好 PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124、torchinfo、matplotlib。

## 0.3 check_env.py 代码逐行讲解

```python
import torch
```
导入 PyTorch 主体。

```python
print("=" * 50)
print("PyTorch 环境自检")
print("=" * 50)
```
打印一个分隔栏,让输出好看点。

```python
print(f"torch 版本:        {torch.__version__}")
```
`torch.__version__` 是个字符串,例如 `'2.6.0+cu124'`。后缀 `+cu124` 告诉你这个 wheel 是为 CUDA 12.4 编译的。

```python
print(f"CUDA 是否可用:     {torch.cuda.is_available()}")
```
最关键的一行。`True` 表示当前 Python 进程能识别到 GPU 且 PyTorch 是 GPU 版本;`False` 就得检查:
- 是不是装的 CPU 版 PyTorch(`+cpu` 后缀)
- 显卡驱动是不是没装
- NVIDIA 控制面板里 GPU 是不是被禁用了

```python
if torch.cuda.is_available():
    print(f"CUDA 版本(运行时): {torch.version.cuda}")
    print(f"GPU 数量:          {torch.cuda.device_count()}")
    print(f"当前 GPU 名称:     {torch.cuda.get_device_name(0)}")
    print(f"GPU 显存:          {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

四个关键查询:
- `torch.version.cuda`:PyTorch 自带的 CUDA runtime 版本(不是驱动支持的最高版本!)
- `torch.cuda.device_count()`:能看到的 GPU 数量
- `torch.cuda.get_device_name(0)`:第 0 块 GPU 的名字
- `torch.cuda.get_device_properties(0).total_memory`:第 0 块 GPU 的显存字节数,除以 `1024**3` 转 GB

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.rand(5, 3, device=device)
print(x)
```
创建一个 `5x3` 的随机 tensor,**直接放到 GPU 上**。如果没 GPU 就降级到 CPU,程序不会崩。

```python
if torch.cuda.is_available():
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    c = a @ b
```
跑一个 `1000x1000 @ 1000x1000` 的矩阵乘法,**真在 GPU 上算**。如果这一步报错或卡住,说明环境是真有问题。

## 0.4 进阶知识

### 0.4.1 为什么我建议装 GPU 版

CPU 和 GPU 跑同一段代码的差距,大矩阵乘法上**经常 50-100x**:

| 操作 | CPU(粗略) | RTX 4060(粗略) |
|------|----------|---------------|
| 1000x1000 matmul | 5 ms | 0.1 ms |
| ResNet50 推理一张图 | 100 ms | 5 ms |
| ResNet50 训练一个 batch(64 张) | 5 s | 0.05 s |

学习阶段小模型 CPU 也能跑,但一旦你想训 CIFAR / 玩 GPT,**没 GPU 几乎无法忍受**。

### 0.4.2 怎么判断装对了

跑完 `check_env.py` 应该看到:

```
torch 版本:        2.6.0+cu124
CUDA 是否可用:     True
CUDA 版本(运行时): 12.4
GPU 数量:          1
当前 GPU 名称:     NVIDIA GeForce RTX 4060 Laptop GPU
GPU 显存:          8.0 GB
```

如果有任一项不对:

| 现象 | 大概率原因 | 修法 |
|------|---------|------|
| `CUDA 是否可用: False` | 装的是 CPU 版 PyTorch | 卸载重装 `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| `CUDA 版本(运行时): None` | 装的是 CPU 版 | 同上 |
| 程序启动直接崩 | 显卡驱动太旧 | 去 NVIDIA 官网下最新驱动 |
| `CUDA error: out of memory` | 显存不够 | 减小 batch size |

### 0.4.3 设备字符串的几种写法

```python
torch.rand(3, device="cpu")              # CPU
torch.rand(3, device="cuda")             # 默认 GPU
torch.rand(3, device="cuda:0")           # 第 0 块 GPU
torch.rand(3, device="cuda:1")           # 第 1 块 GPU
```

学习阶段基本只用 `"cuda"` / `"cpu"`,多卡部署才会用到下标。

### 0.4.4 一个常被忽略的细节:NVML vs CUDA runtime

`nvidia-smi` 用的是 **NVML**(NVIDIA Management Library),看的是**驱动支持的最高 CUDA**。
`torch.version.cuda` 是 **CUDA runtime 版本**(PyTorch 自带的那个)。

这两个不一致完全正常。本机 NVML 说 13.1,runtime 说 12.4,只要 `12.4 ≤ 13.1` 就能跑。

## 0.5 常见坑

1. **conda 装 PyTorch** 和 **pip 装 PyTorch** 混用 → 容易出现两个 PyTorch,`import torch` 不知道进哪个。建议**只用 pip / uv**,不用 conda 装 PyTorch。
2. **没激活 venv 就跑** → 用的是系统 Python,`import torch` ModuleNotFoundError。PowerShell 提示符前有 `(.venv)` 才算激活。
3. **多块 GPU 机器上 PyTorch 默认只看到部分** → `CUDA_VISIBLE_DEVICES=0,1` 环境变量控制可见的卡。
4. **在 Jupyter 里 `!pip install` 装到了别的 Python** → 永远先确认 `!which python` 和 `!python -c "import sys; print(sys.executable)"` 再装。

## 0.6 下一步

环境没问题的话,进入 [`01-tensor-basics/doc/01-tensor-basics.md`](../../01-tensor-basics/doc/01-tensor-basics.md) 开始学张量。
