"""03-3 列表型 Dataset(数据从硬盘读)

教学场景:数据是图片路径列表 + 标签列表,__getitem__ 临时读盘。
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ImageListDataset(Dataset):
    """从路径列表读图片的 Dataset"""

    def __init__(self, image_paths, labels, transform=None):
        self.paths = image_paths
        self.labels = labels
        self.transform = transform  # 可选的预处理函数

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # 临时读图(不一次性全读进内存)
        img = Image.open(self.paths[i]).convert("RGB")
        label = self.labels[i]

        if self.transform is not None:
            img = self.transform(img)  # 转 tensor / 归一化

        return img, label


# --- 模拟数据 ---
# 实际场景:os.listdir() 读文件夹,这里用 fake 路径演示结构
fake_paths = [f"img_{i:04d}.jpg" for i in range(50)]
fake_labels = [i % 10 for i in range(50)]  # 10 分类

ds = ImageListDataset(fake_paths, fake_labels)
print(f"数据集大小: {len(ds)}")

# 注意:fake_paths 不是真文件,真跑会 FileNotFoundError
# 实际使用时:
# 1. 先把图片放到 data/ 下
# 2. ImageListDataset(real_paths, real_labels, transform=...)
# 3. DataLoader(ds, batch_size=32, num_workers=2, shuffle=True)
