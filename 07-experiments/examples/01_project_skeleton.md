# 07-1 PyTorch 项目骨架(推荐布局)

```
07-experiments/
├── models/
│   ├── __init__.py
│   ├── simple_cnn.py        # 自己写的网络
│   └── resnet_finetune.py   # torchvision 网络改的
├── utils/
│   ├── __init__.py
│   ├── train.py             # 一个 epoch 的训练 + 验证
│   ├── evaluate.py          # 测试集评估 + 指标
│   └── visualize.py         # 训练曲线 / 混淆矩阵
├── data/                    # 数据集缓存(自动生成)
├── checkpoints/             # 模型权重
├── outputs/                 # 训练曲线 / 报告
├── configs/
│   └── default.yaml         # 超参(可放路径/lr/batch_size/epochs)
├── train.py                 # 入口:train + val + 保存
└── evaluate.py              # 入口:加载权重 + 测试
```

## 关键习惯

- **可复现性**:`torch.manual_seed` / `np.random.seed` / `random.seed` 三件套
- **配置外置**:超参放 YAML 或 argparse,不写死在代码里
- **可观测性**:每个 epoch 打印 train_loss / val_loss / val_acc
- **最佳权重保存**:`val_acc` 提升才覆盖 `best.pt`,最后用 best.pt 测
- **断点续训**:`optimizer.state_dict` + `epoch` + `best_acc` 都存
