"""03-4 transforms 流水线

图片 → tensor → 归一化,这是图像任务的标配预处理。
"""
from torchvision import transforms

# 单步操作
resize = transforms.Resize((32, 32))           # 缩到 32x32
to_tensor = transforms.ToTensor()               # PIL/numpy (HxWxC, 0-255) → tensor (CxHxW, 0-1)
normalize = transforms.Normalize(               # 用 ImageNet 均值/方差归一化
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

# 串成流水线(顺序很重要!)
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),  # 数据增强
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 验证集一般不做随机增强
val_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 演示一下作用对象
from PIL import Image
import numpy as np

# 造一个 100x100 的假图
fake_img = Image.fromarray(
    (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
)

x = train_transform(fake_img)
print(f"原图 PIL  size: {fake_img.size}")           # (100, 100)
print(f"变换后 shape:  {x.shape}")                    # torch.Size([3, 32, 32])
print(f"数据类型:      {x.dtype}")                    # torch.float32
print(f"数值范围:      [{x.min():.3f}, {x.max():.3f}]")  # 归一化后会超出 0-1

# 关键点:
# - PIL/numpy 是 HxWxC,转 tensor 后是 CxHxW(PyTorch 约定)
# - 像素从 0-255 变 0-1(ToTensor)
# - 归一化后均值约 0、标准差约 1(对训练稳定很重要)
