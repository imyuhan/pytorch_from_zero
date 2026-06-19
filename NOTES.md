# PyTorch 学习笔记

## 00-setup / 环境验证

- 显卡: NVIDIA GeForce RTX 4060 笔记本版, 8G 显存
- 驱动: 591.74, 支持 CUDA 最高 13.1
- PyTorch wheel: cu124(自带的 CUDA 12.4 runtime,够用)
- `torch.cuda.is_available()` → True
- `torch.cuda.get_device_name(0)` → NVIDIA GeForce RTX 4060 Laptop GPU

## 01-tensor-basics / Tensor 内存:逻辑大小 vs 实际存储

PyTorch 的 tensor 其实是**两层**的:
- **Tensor 层**(元数据): `shape` / `dtype` / `stride`,描述"怎么解读下面那块内存"
- **Storage 层**(原始字节): 一段连续内存,真正存数据的地方

`numel() * element_size()` 和 `untyped_storage().nbytes()` 的区别就在这:

| 调用 | 看的层 | 性质 |
|---|---|---|
| `numel() * element_size()` | Tensor 层 | 按逻辑结构**推算**的大小 |
| `untyped_storage().nbytes()` | Storage 层 | PyTorch **实际分配**的字节数 |

原生 dense + contiguous 的 tensor(像 `torch.ones(3, 4)`)两者必然相等;
但只要是 view / 切片 / 转置,两者就会分家——因为 view 共享原 tensor 的 storage。

```python
x = torch.arange(12)                 # int64, 12 元素 = 96 字节
y = x[:6]                             # 切片:逻辑只剩 6 元素,但 storage 还是 x 的那块
y.numel() * y.element_size()          # 48  (逻辑值)
y.untyped_storage().nbytes()          # 96  (实际值,共享了 x 的 storage)
```

**验证套路**:练习里估算 `torch.ones(3, 4)` 占 48 字节(12 × 4),用
`untyped_storage().nbytes()` 验证一下,数字对得上就说明是个干净的 dense tensor。
哪天这俩数对不上,就是 PyTorch 在提示"这玩意是 view,共享着别人的内存"——改它会动到原 tensor。

**易混点**:`tensor.nbytes`(无 `untyped_storage()`)是个属性,等价于
`numel() * element_size()`,**不是**实际 storage 大小。名字像但别搞混:

```python
x1 = torch.ones(3, 4)
x1.nbytes                          # 48 = 12 * 4  (逻辑)
x1.untyped_storage().nbytes()      # 48             (实际,本例相等)
```

## 01-02-cpu-vs-gpu / 计时:CPU vs GPU 耗时怎么测

测运算耗时,CPU 和 GPU 是两套写法,**核心区别就一句**:GPU 操作是异步的,
测之前必须 `torch.cuda.synchronize()`,否则你拿到的是"提交耗时"不是"算完耗时"。

### 1. CPU 计时 — `time.perf_counter()`

```python
import time

start = time.perf_counter()
# ... CPU 运算 ...
end = time.perf_counter()

print(f"耗时: {end - start:.4f} 秒")
```

为什么用 `perf_counter()` 而不是 `time.time()`?
- `perf_counter()` 是**单调时钟**(只增不减),精度更高(微秒级)
- `time.time()` 是壁钟时间,系统时间被改一下就会乱
- 测耗时一律推荐 `perf_counter()`

### 2. GPU 计时 — 必须加 `torch.cuda.synchronize()`

```python
import torch
import time

# 关键:测之前先等 GPU 排空上一轮
torch.cuda.synchronize()

start = time.perf_counter()
# ... GPU 运算 (kernel 提交到 CUDA 流就立刻返回) ...
torch.cuda.synchronize()   # 这里等 GPU 真的算完

end = time.perf_counter()
print(f"耗时: {end - start:.4f} 秒")
```

不加 sync 会怎样?CUDA kernel 是丢到流里就异步返回的,你测到的
"耗时"其实是 CPU 提交命令的调度开销——会得到一个**虚假的"GPU 巨快"**
的结果,差几个数量级都有可能。

### 一张表看清区别

| 场景 | 计时 API | 要不要 `cuda.synchronize()` |
|---|---|---|
| 纯 CPU / Python 代码 | `time.perf_counter()` | 不需要 |
| PyTorch GPU 张量运算 | `time.perf_counter()` | **必须**,前后各一次 |
| 比 CPU vs GPU 同一运算 | 同一套 `perf_counter` | GPU 端必须有 sync 才公平 |

### 计时三坑(实测踩过)

光 sync 正确还不够,这三件事一起决定了"测出来的加速比到底是不是真的":

| 坑 | 现象 | 解决 |
|---|---|---|
| **矩阵太小** | 1000×1000 时 CPU 可能比 GPU 还快 | 至少用 2048×2048 以上 |
| **没预热** | 第一次 GPU 调用含 CUDA 冷启动,慢 10~50 倍 | 先空跑 5~10 次再计时 |
| **单次测量** | 调度 / cache miss 噪声大 | 跑 50 次取平均 |

经验阈值(在 RTX 4060 上):

| 尺寸 | CPU vs GPU |
|---|---|
| < 512×512 | CPU 更快(GPU 启动开销 > 计算本身) |
| ~1000×1000 | 胶着(单次测常看到 CPU 赢) |
| 2048×2048 | GPU 开始领先,约 5~10x |
| 4096×4096+ | GPU 碾压,10x+ |

**结论**:看到"GPU 比 CPU 还慢"先别怀疑 GPU,大概率是矩阵太小 / 没预热 / 单次测。三者必居其一。
