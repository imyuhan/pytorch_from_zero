# PyTorch 学习笔记

> 从零开始学 PyTorch 的练习项目。每个模块按 **概念 → 例子 → 练习** 的节奏走,先看 `examples/` 跑通,再做 `exercises/` 巩固。

## 环境

| 项 | 值 |
|------|------|
| Python | 3.12.13 |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| GPU | NVIDIA GeForce RTX 4060 笔记本(8 GB 显存) |
| CUDA 运行时 | 12.4 |
| 包管理 | uv / pip(依赖写在 [`pyproject.toml`](pyproject.toml)) |
| 虚拟环境 | `.venv/` |

## 快速开始

```powershell
# 1. 建虚拟环境 + 装依赖(uv 方式,推荐)
uv venv
uv pip install -e .

# 或者用传统 pip
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .

# 2. 验证环境(应输出 torch 版本 + CUDA 可用 + GPU 名称)
python 00-setup\check_env.py
```

Linux / macOS:

```bash
# uv
uv venv
uv pip install -e .

# 或传统 pip
source .venv/bin/activate
pip install -e .

python 00-setup/check_env.py
```

## 课程结构

| # | 目录 | 主题 | 例子 | 练习 |
|:-:|------|------|:----:|:----:|
| 0 | `00-setup/` | 环境验证 | 1 | – |
| 1 | `01-tensor-basics/` | 张量基础(创建、索引、设备迁移、形状、内存) | 6 | 5 |
| 2 | `02-autograd/` | 自动求导(requires_grad / backward / no_grad / 手写 GD) | 5 | 5 |
| 3 | `05-dataset-dataloader/` | Dataset / DataLoader / transforms | 6 | 5 |
| 4 | `03-attention/` | 维度变换 / 手写 scaled dot-product attention / 多头拆合头 / 与官方 SDPA 对齐 | 3 | 3 |
| 5 | `06-pretrained-models/` | torchvision 预训练模型 / 推理 / 替换头 / 冻 backbone | 5 | 4 |
| 6 | `07-cnn/` | nn.Module / 卷积 / 池化 / 训练 / 验证 / 保存加载 | 7 | 5 |
| 7 | `08-experiments/` | 工程模板 / 训练 MNIST / CIFAR-10 / ResNet 微调 | 5 | 4 |

每个模块的细节大纲(关键词、概念、目录导航)看对应模块的 `README.md`。

每个模块的**详细教学文档**放在 `doc/<模块名>.md`,包含:
- 基础概念(名词解释、API 详解)
- 进阶知识(底层原理、工程技巧)
- 所有 `examples/` 脚本的逐行讲解
- 常见坑 + 学习自检

## 📖 教学文档导航

| 文档 | 链接 |
|------|------|
| 00 环境验证 | [`00-setup/doc/00-setup.md`](00-setup/doc/00-setup.md) |
| 01 张量基础 | [`01-tensor-basics/doc/01-tensor-basics.md`](01-tensor-basics/doc/01-tensor-basics.md) |
| 02 自动求导 | [`02-autograd/doc/02-autograd.md`](02-autograd/doc/02-autograd.md) |
| 03 数据加载 | [`05-dataset-dataloader/doc/05-dataset-dataloader.md`](05-dataset-dataloader/doc/05-dataset-dataloader.md) |
| 04 维度变换与注意力 | [`03-attention/doc/03-attention.md`](03-attention/doc/03-attention.md) |
| 05 预训练模型 | [`06-pretrained-models/doc/06-pretrained-models.md`](06-pretrained-models/doc/06-pretrained-models.md) |
| 06 卷积神经网络 | [`07-cnn/doc/07-cnn.md`](07-cnn/doc/07-cnn.md) |
| 07 综合实验 | [`08-experiments/doc/08-experiments.md`](08-experiments/doc/08-experiments.md) |

## 📄 论文阅读

学习过程中收集的论文 / 资料,以**自包含的中文笔记**形式放在 [`papers/`](papers/README.md) 下。

| 论文 | 关联章节 |
|------|---------|
| [Attention Is All You Need (Vaswani et al., NeurIPS 2017)](papers/Attention_Is_All_You_Need_CN.md) | [03-attention](03-attention/README.md) |

新增论文的约定:见 [`papers/README.md`](papers/README.md)。

## 学习路径建议

```text
01 张量基础 → 02 自动求导 → 03 数据加载 → 04 维度变换与注意力
                                              ↓
                07 综合实验 ← 06 CNN  ← 05 预训练模型
```

- **01 + 02** 是底子(必看),后面所有东西都靠它们
- **03** 把数据和代码解耦,从此训练用真实数据集
- **04** 把 01 章的 QKV shape 流补成完整注意力公式 + 多头 + 官方 SDPA 对齐
- **05** 用别人训好的模型,先看到效果
- **06** 自己写一个 CNN,理解 backbone 怎么搭
- **07** 把前面所有章节全部串成完整工程

## 跑例子

每个模块的 `examples/` 都可以独立跑。约定:任何 `python` 命令都先 `cd` 到项目根目录。

```powershell
# 跑单个例子
.venv\Scripts\python.exe 01-tensor-basics\examples\01_create_and_check.py

# 跑一个模块下所有 examples(简单循环,本项目没有提供统一入口)
```

## 仓库结构

```
.
├── .venv/                     # 虚拟环境(已 git ignore)
├── 00-setup/                  # 环境验证
│   ├── check_env.py
│   ├── doc/00-setup.md        # 教学文档
│   └── README.md
├── 01-tensor-basics/          # 张量
│   ├── examples/              # 6 个可跑例子
│   ├── exercises/             # 5 个练习(留空待你写)
│   ├── doc/01-tensor-basics.md  # 教学文档
│   └── README.md              # 概念 + 章节大纲
├── 02-autograd/               # 自动求导
│   ├── examples/              # 5 个
│   ├── exercises/             # 5 个
│   ├── doc/02-autograd.md     # 教学文档
│   └── README.md
├── 05-dataset-dataloader/     # 数据加载
│   ├── examples/              # 6 个
│   ├── exercises/             # 5 个
│   ├── doc/05-dataset-dataloader.md
│   └── README.md
├── 03-attention/              # 维度变换与注意力矩阵
│   ├── examples/              # 3 个(02 端到端 demo + 03 mask 对比 + 04 batch 对比)
│   ├── exercises/             # 3 个(手写 MHA / causal mask / 官方 SDPA 重写)
│   ├── src/attention_tools.py # 课堂演示工具库
│   ├── data/lesson_data.json  # 3 条手写小样本
│   ├── doc/03-attention.md
│   └── README.md
├── 06-pretrained-models/      # 预训练模型
│   ├── examples/              # 5 个
│   ├── exercises/             # 4 个
│   ├── doc/06-pretrained-models.md
│   └── README.md
├── 07-cnn/                    # 卷积神经网络
│   ├── examples/              # 7 个
│   ├── exercises/             # 5 个
│   ├── doc/07-cnn.md          # 教学文档
│   └── README.md
├── 08-experiments/            # 综合实验
│   ├── examples/              # 5 个
│   ├── exercises/             # 4 个
│   ├── doc/08-experiments.md
│   └── README.md
├── data/                      # 数据集缓存(MNIST / CIFAR-10)
├── checkpoints/               # 训练保存的模型权重
├── notebooks/                 # Jupyter 实验
├── papers/                    # 论文阅读笔记(中文)
├── NOTES.md                   # 我的随手笔记
├── pyproject.toml             # 依赖清单(uv / pip)
├── .gitignore
└── README.md                  # ← 你正在看
```

## 一些约定

- **代码注释中文 / 标识符英文**(配套文档中文,代码风格沿用 Python 社区约定)
- **commit message 风格**:`type 前缀英文 + 中文描述`(例如 `docs: 更新 README`)
- **每个 example 都标了对应大纲序号**,做练习时知道它在考哪个点
- **不写"截图"等指向原始输入材料的描述**,所有文字都是自包含的
- 例子代码优先 **在 GPU 上跑**,但加了 `device` 兼容,没 GPU 也能降级到 CPU(只是慢)

## 许可证

[MIT](LICENSE)
