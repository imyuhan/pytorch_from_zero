# 01 - 张量基础(教学文档)

> 本节覆盖 PyTorch 的"原子单位"—— Tensor。学完你应该能**口述 14 条学习目标**并熟练使用 `torch.tensor` / `torch.ones` / `torch.zeros` / `torch.randn` 等核心 API。

## 1.1 涉及的文件

| 文件 | 主题 |
|------|------|
| `examples/01_create_and_check.py` | 4 种创建方式 + 6 个属性 |
| `examples/02_device_move.py` | `.to(device)` 设备迁移 |
| `examples/03_numpy_interop.py` | NumPy ↔ Tensor 互转 |
| `examples/04_shape_ops.py` | squeeze / unsqueeze / view / reshape / contiguous / cat / stack |
| `examples/05_transformer_qkv.py` | 3 维张量 + 矩阵乘法 + 广播 |
| `examples/06_index_and_memory.py` | 索引切片 + 内存共享关系 |

## 1.2 基础知识

### 1.2.1 Tensor 是什么

**Tensor = 多维数组 + 元数据**(dtype、device、shape、requires_grad)。从 0 维到任意维:

| 维度 | 数学名 | 例子 |
|:----:|------|------|
| 0 | 标量(scalar) | `torch.tensor(3.14)` |
| 1 | 向量(vector) | `torch.tensor([1, 2, 3])`,shape=`(3,)` |
| 2 | 矩阵(matrix) | `torch.rand(3, 4)`,shape=`(3, 4)` |
| 3 | 批次张量 | `torch.rand(B, S, H)`,shape=`(B, S, H)` |
| 4+ | 图像 / 视频 | `torch.rand(B, C, H, W)`,shape=`(B, 3, 224, 224)` |

注意:PyTorch 跟 NumPy 在维度顺序上**习惯相反** ——

- **NumPy / PIL**: `(H, W, C)`,高维用 `HWC` 在前
- **PyTorch**: `(C, H, W)`,通道在**前**(channels-first)

这就是为什么图片读进来后**一定要 `permute(2, 0, 1)`** 或者用 `transforms.ToTensor()` 帮你转。

### 1.2.2 dtype(元素类型)

| dtype | 字节 | 范围 | 用途 |
|-------|:----:|------|------|
| `torch.float32` / `float` | 4 | 通用浮点 | 99% 的场景 |
| `torch.float16` / `half` | 2 | 范围小,精度低 | 混合精度训练 |
| `torch.bfloat16` | 2 | 范围大,精度低 | 深度学习专用 |
| `torch.float64` / `double` | 8 | 高精度 | 数值计算 |
| `torch.int8` ~ `int64` | 1/2/4/8 | 整数 | 标签、索引 |
| `torch.bool` | 1 | True/False | 掩码 |

**默认规则**:
- `torch.tensor([1, 2, 3])` → `int64`(整数 Python list 默认推成 int)
- `torch.tensor([1.0, 2.0])` → `float32`
- `torch.randn(...)` / `zeros` / `ones` → `float32`

### 1.2.3 device(所在设备)

两个值:`cpu` 或 `cuda:0`。CPU 是默认;GPU 需要先 `torch.cuda.is_available()` 检查,没装 CUDA 版的 PyTorch 就用不了。

### 1.2.4 shape 与 size

```python
x = torch.zeros(2, 3, 4)
x.shape        # torch.Size([2, 3, 4])  ← 习惯用这个
x.size()       # torch.Size([2, 3, 4])  ← 等价
x.size(0)      # 2                     ← 拿到具体某一维
x.numel()      # 24                    ← 总元素数 = 2*3*4
x.element_size()  # 4                  ← 单个元素的字节数(float32)
```

`torch.Size` 本质是 `tuple` 的子类,可以 `x.shape[0]` / `len(x.shape)`,也可以解包 `b, c, h, w = x.shape`。

### 1.2.5 requires_grad

`requires_grad=True` 是 autograd 的开关,**打开后 PyTorch 会建计算图**,所有作用于这个 tensor 的运算都会被记录。后面 02 节详细讲。

### 1.2.6 内存占用估算

```python
num_bytes = tensor.numel() * tensor.element_size()
```

- `numel()` = 总元素数
- `element_size()` = 单个元素字节数
- 乘积就是 tensor 实际占用的**裸内存**(不含 Python 对象的开销)

例:`torch.zeros(1024, 1024, dtype=torch.float32)` → `1024*1024*4 = 4 MB`。

## 1.3 逐个 example 讲解

### 1.3.1 `01_create_and_check.py` — 创建与检查

```python
a = torch.tensor([[1, 2], [3, 4]])      # 从 Python list
b = torch.ones(2, 3)                     # 全 1
c = torch.zeros(2, 3)                    # 全 0
d = torch.randn(2, 3)                    # N(0, 1) 随机
```

**注意**:
- `torch.tensor(list)` 会**复制** Python 列表的数据,不受 list 后续修改影响
- `torch.ones(2, 3)` 第一个参数是**行数**?不,是**第一维的大小**!`torch.ones(2, 3)` 实际形状是 `(2, 3)` = 2 行 3 列
- `torch.randn` 跟 `torch.rand` 的区别:`rand` 是 `[0, 1)` 均匀分布,`randn` 是 `N(0, 1)` 正态分布

```python
print("shape:        ", a.shape)         # torch.Size([2, 2])
print("dtype:        ", a.dtype)         # torch.int64
print("device:       ", a.device)        # cpu
print("requires_grad:", a.requires_grad) # False
print("numel:        ", a.numel())       # 4
print("element_size: ", a.element_size())  # 8 字节(int64)
```

- `dtype=int64` 是因为 `torch.tensor([[1, 2], [3, 4]])` 传入的是 Python `int`,PyTorch 默认推成 int64
- `numel() * element_size() = 4 * 8 = 32 字节`

### 1.3.2 `02_device_move.py` — 设备迁移

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(3, 3)
x_gpu = x.to(device)
print("x 在哪:       ", x.device)        # cpu
print("x_gpu 在哪:   ", x_gpu.device)    # cuda:0
print("是同一对象?", x is x_gpu)         # False
```

**核心要点**:
1. `.to()` **返回新 tensor**,原 tensor 不动(`x` 还在 CPU)
2. `.to()` 是个**智能方法**:
   - `.to(device)` → 移动设备
   - `.to(torch.float32)` → 转换 dtype
   - `.to(device="cuda", dtype=torch.float16)` → 同时改两样
3. **原 tensor 还在老位置**,需要的话手动用 `x = x.to(device)` 覆盖

```python
torch.cuda.memory_allocated()       # 当前已分配
torch.cuda.max_memory_allocated()   # 峰值
torch.cuda.empty_cache()            # 释放缓存(不释放你引用的张量)
```

- `memory_allocated()` 返回**字节数**,要算 MB 就除 `1024**2`
- `empty_cache()` 把"PyTorch 缓存池里空闲的块"还给 GPU,**不释放你还引用的张量**。平时不用调,在你看到 `out of memory` 但其实显存够用时才有用。

### 1.3.3 `03_numpy_interop.py` — NumPy 互转

```python
np_arr = np.array([1.0, 2.0, 3.0])
t_shared = torch.from_numpy(np_arr)   # 共享内存
t_copy   = torch.tensor(np_arr)       # 复制
```

**这是 PyTorch 最常踩的坑之一**:
- `torch.from_numpy(numpy_arr)`:**共享底层内存**,改 `np_arr[0]` 会同步反映到 `t_shared` 上
- `torch.tensor(numpy_arr)`:先复制,改 `numpy_arr` 不影响 `t_copy`

共享内存能省内存、提升性能(预处理到训练一步到位),但**任何一边的 in-place 修改都会污染另一边**,debug 时容易抓狂。

```python
gpu_t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
np_back = gpu_t.cpu().numpy()
```

**GPU tensor 不能直接 `.numpy()`**,会报:
```
TypeError: can't convert cuda:0 device type tensor to numpy
```

正确做法:`.cpu().numpy()`,先把数据搬到 CPU 再转 NumPy。

### 1.3.4 `04_shape_ops.py` — 形状操作

#### squeeze / unsqueeze

```python
x = torch.randn(3, 4)            # (3, 4)
x_unsq = x.unsqueeze(0)          # (1, 3, 4) — 在第 0 维插 1
x_sq = x_unsq.squeeze(0)         # (3, 4)    — 删掉所有为 1 的维
```

`unsqueeze(dim)` 在第 `dim` 维插入一个大小为 1 的维度;`squeeze(dim=None)` 默认删所有大小为 1 的维度,传 `dim` 时只删那一个。

**实际用处**:广播对齐。例如做 batch matmul 需要 `(B, S, H) @ (H, H)`,但手头只有 `(S, H)`,`unsqueeze(0)` 一下就成 `(1, S, H)`,广播自动生效。

#### view / reshape / contiguous
Pytorch的tensor有两个独立的属性：

| 属性         | 含义                 | 谁控制            |
|------------|:-------------------|----------------|
| shape(形状)  | 逻辑上"看上去"是几维几行      | .shape/.size() |
| stride(步长) | 物理内存里某一维移动1步要跨多少元素 | .stride()      |

`is_contiguous()`是检查stride排布是不是和shape一致（具体地说，沿着每一维前进1步，元素在内存里是不是“连续无缝地”挨着）。

`contiguous()`返回一个连续的版本，如果本来就是连续的直接返回self，如果不连续拷贝一份新内存，重新排列成连续布局再返回
```python
a = torch.arange(12)              # (12,)
b = a.view(3, 4)                  # (3, 4)  — 共享内存,要求 a 连续
c = a.reshape(3, 4)               # (3, 4)  — 尽量共享,不行就复制
```

`view` 和 `reshape` 都能改变形状,区别:
- `view`:要求 tensor **内存连续**,满足就共享,不满足就**报错**
- `reshape`:优先共享,不能共享时**自动复制**(更友好)

```python
mat = torch.arange(6).reshape(2, 3)  # (2, 3), 连续
trans = mat.t()                       # (3, 2), 不连续!
trans.is_contiguous()                 # False
trans_c = trans.contiguous()          # 复制一份成连续
```

**`.t()`(转置)之后不连续**,因为内存布局变成"按列"而不是"按行"了。这时:
- `trans.view(3, 2)` → **报错**
- `trans.reshape(3, 2)` → 默默复制一份,正常返回
- `trans.contiguous().view(3, 2)` → 显式复制再 view

#### cat vs stack

```python
t1 = torch.tensor([[1, 2], [3, 4]])   # (2, 2)
t2 = torch.tensor([[5, 6], [7, 8]])   # (2, 2)

torch.cat([t1, t2], dim=0)            # (4, 2) — 沿已有维度拼
torch.stack([t1, t2], dim=0)          # (2, 2, 2) — 新增一个维度堆
```

- `cat`:沿**已有**的某一维**拼接**(像把两块布缝一起)
- `stack`:沿**新的**一维**堆叠**(像把两块布叠成两层)

口诀:concat 是**加长**,stack 是**加厚**。

### 1.3.5 `05_transformer_qkv.py` — 3 维张量 + QKV + 广播

这一节是为后面学 Transformer 铺垫的实战。它看着吓人,但拆开就是三个独立小知识点:**3 维张量的"批次矩阵乘法"**、**Q/K/V 的角色分工**、**广播机制**。一个一个来。

#### (1) 先搞懂输入 `x` 的形状

```python
torch.manual_seed(42)               # 钉死随机种子,每次跑结果一样
B, S, H = 2, 4, 8                   # Batch=2, Seq_len=4, Hidden=8
x = torch.randn(B, S, H)            # (2, 4, 8)
```

`x` 是一个 **3 维张量**,三个维度各代表一个语义层:

| 维度 | 名字 | 这里等于 | 含义 |
|:----:|------|:----:|------|
| 第 0 维 | Batch(B) | 2 | 一次处理 2 个独立样本(2 句话) |
| 第 1 维 | Sequence(S) | 4 | 每个样本有 4 个 token(每句 4 个词) |
| 第 2 维 | Hidden(H) | 8 | 每个 token 用 8 维向量表示 |

可以把它想象成 **"2 句话 × 4 个词 × 8 维数字"** 的一个表格。NLP 里几乎所有输入都是 `(B, S, H)` 这个形状 —— 你后面会反复见到。

> 关于 `torch.manual_seed(42)`:它是 PyTorch 的**随机数种子**。不设的话,每次 `torch.randn` 抽到的数都不一样;设了之后,不管谁跑、不管在哪儿跑,**抽到的随机数序列完全相同**。教学、debug、写报告时几乎必须固定。42 只是任意挑的一个数,你写 0、2024、666 都行,关键是**固定**。

#### (2) Q、K、V 到底是什么?为什么是三个?

代码里:

```python
W_q = torch.randn(H, H)   # (8, 8)  ← "提问"用的投影矩阵
W_k = torch.randn(H, H)   # (8, 8)  ← "被查"用的投影矩阵
W_v = torch.randn(H, H)   # (8, 8)  ← "实际内容"用的投影矩阵

Q = x @ W_q   # (B, S, H)
K = x @ W_k
V = x @ W_v
```

**关键事实**:Q、K、V **是同一个输入 `x` 经过三个不同的权重矩阵投影出来的三份"分身"**。它们一开始内容长得差不多,只是**角色不同**。

这个名字是借用了图书馆查资料的比喻:

| 名字 | 全称 | 这个向量的"身份" | 类比 |
|------|------|------------------|------|
| **Q** | Query(查询) | 主动问问题:"**我**想找跟**我**相关的词" | 你的问题 |
| **K** | Key(键) | 被动提供信息:"**我**的标签是什么,可以拿来跟我匹配" | 书脊上的索引 |
| **V** | Value(值) | 真正的内容:"**我**实际要传递给别人的信息" | 书里面的内容 |

> 算 Q·K 决定要不要"翻"这本书,翻了之后拿走的是 V。

**为什么不共用一个 `W`?**

因为 Q、K、V 的"职责"不一样 —— 让网络学出来**三个不同的投影矩阵**,才能让"提问"和"被查"和"提供内容"在三个不同的子空间里各司其职。如果共用一个,模型表达能力会差很多。

#### (3) `(B, S, H) @ (H, H) → (B, S, H)` 的魔法

`Q = x @ W_q` 这一行,左 `(2, 4, 8)`、右 `(8, 8)`,按线性代数来说两边都对不上 —— 怎么就乘起来了?

**PyTorch 的 `@` 算子有一条隐藏规则**:**只对最后两维做矩阵乘法,前面的维度保持不动**(只要满足广播规则)。

等价于这个 `for` 循环,但不用你写:

```python
for b in range(B):
    Q[b] = x[b] @ W_q   # 把 2 句话一句一句算
```

GPU 跑这种批处理比 CPU 的 `for` 循环快几十到上百倍 —— 这就是**为什么要把数据堆成 batch** 的根本原因。

#### (4) `Q @ K^T` 的意义:两两相关度

```python
K_t = K.transpose(-2, -1)   # (B, S, H) → (B, H, S),最后两维互换
scores = Q @ K_t             # (B, S, H) @ (B, H, S) → (B, S, S)
```

`scores` 形状是 `(2, 4, 4)`,即 **"每个样本里任意两个 token 之间两两的相关分数"**,是个 `S × S` 的方阵。

具体到第 2 个 token(`S` 维度索引=2)这一行,它有 4 个数,分别表示**这个 token 作为"查询者"**,跟同句里 4 个 token(包括它自己)分别有多相关。两个向量点积(对应位置相乘再求和)越大 = 越相关。

> 实际 Transformer 里 `scores` 还会除以 `sqrt(H)` 再 `softmax`,这里简化了。

#### (5) `scores @ V` 的意义:加权汇总

```python
attn = scores @ V   # (B, S, S) @ (B, S, H) → (B, S, H)
```

这一句把"对所有 token 的关注度"**加权汇总成新表示**。假设"我"(第 2 个 token)70% 关注自己、20% 关注第 0 个、10% 关注第 3 个,那我的新向量 = `0.7·V[我] + 0.2·V[0] + 0.1·V[3]`。

形状从 `(B, S, S)` 又回到 `(B, S, H)`:每个 token 现在都是一个"看完全句后"的表示。

#### (6) 完整流程图

```
输入 x (B, S, H)            ← 2 句话 × 4 词 × 8 维
   │
   ├──→ @ W_q ─→ Q (B, S, H)   ← "我是谁,我想问什么"
   ├──→ @ W_k ─→ K (B, S, H)   ← "我的标签"
   └──→ @ W_v ─→ V (B, S, H)   ← "我的实际内容"
              ↓
   K 转置最后两维 → K_t (B, H, S)
              ↓
        scores = Q @ K_t  → (B, S, S)   ← 两两关联分数
              ↓
        attn = scores @ V → (B, S, H)   ← 加权汇总后的输出
```

#### (7) 广播(Broadcasting)

文件最后:

```python
a = torch.ones(4, 4)
b = torch.tensor([1., 2., 3., 4.])     # (4,)
(a / b).shape                            # (4, 4)
```

`(4, 4) / (4,)` 怎么除?两边形状不一样 —— PyTorch 自动按规则**把小 tensor "拉"到大 tensor 的形状**,再做逐元素除法。

**广播的三条规则**(从右往左对齐维度):
1. 每个维度要么**相等**
2. 要么其中一个是 **1**(会自动"扩展"成匹配的那个)
3. 要么**不存在**(维度数少的会在**左侧补 1**)

这个例子走一遍:
- `a` 形状 `(4, 4)`,`b` 形状 `(4,)`
- 对齐到右边:`a=(4, 4)`,`b=(1, 4)`(在左边补 1)
- 第 1 维:4 vs 4,相等 ✓
- 第 0 维:4 vs 1,`b` 那维是 1 → 自动扩展成 4
- 结果:逐元素除,形状 `(4, 4)`

**不能广播的情况**:某个维度既不相等也不为 1,比如 `(3, 4)` 和 `(3,)` —— 对齐成 `(3, 4)` 和 `(1, 3)`,最右维 4 ≠ 3 且 ≠ 1 → 直接报错 `RuntimeError`。

**广播的实际用处**:
- 写公式时不用手动复制(节省内存)
- 跟 `unsqueeze` 配合做对齐(在 1.3.4 提过)
- 几乎所有"加偏置"、"按行/按列归一化"的写法都靠它

#### 补充:`transpose(-2, -1)` 跟 `.t()` 的区别

| 操作 | 适用范围 | 例子 |
|------|---------|------|
| `.t()` | **只对 2 维**张量生效 | `(3, 4).t()` → `(4, 3)` |
| `.transpose(dim0, dim1)` | 任意维度,**交换指定两维** | `(B, S, H).transpose(-2, -1)` → `(B, H, S)` |
| `.permute(d0, d1, d2, ...)` | 任意维度,**整体重排所有维** | `(B, C, H, W).permute(0, 2, 3, 1)` → `(B, H, W, C)` |

代码里用 `transpose(-2, -1)` 是因为 `K` 是 3 维张量,`.t()` 会报错。`-2, -1` 这种"倒数索引"在维度数不确定时特别好用。

### 1.3.6 `06_index_and_memory.py` — 索引切片 + 内存关系

#### 索引语法

```python
x = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

x[0]           # 第 0 行 → shape (4,)
x[:, 0]        # 第 0 列 → shape (3,)
x[1, 2]        # 单个值 → 0-dim tensor(用 .item() 拿 Python 数)
x[0:2, 1:3]    # 切片 → shape (2, 2)
```

`.item()` 用来把 0-dim tensor 拆成 Python 数字,在 print 损失值、做早停判断时常用。

#### 内存关系(本节最易踩坑)

```python
a = torch.arange(6)            # (6,)
b = a.view(2, 3)                # 共享内存!
c = a.reshape(2, 3)             # 优先共享
d = a.clone()                   # 强制复制,完全独立
e = a.t().contiguous()          # 转置后不连续,contiguous 复制成连续

b[0, 0] = 999
# 此时 a[0] 也是 999(view 共享)
d[0] = 888
# a[0] 不变(独立)
```

**`view` 不复制,`reshape` 看心情,`clone` 必复制,`contiguous` 不连续时复制**。一张表记清:

| 操作 | 复制吗? |
|------|------|
| `view` | ❌ 不复制,要求连续 |
| `reshape` | 能不复制就不复制,不行就复制 |
| `clone` | ✅ 总是复制 |
| `contiguous` | 已经连续就不复制,否则复制 |
| `transpose` / `permute` | ❌ 不复制,只改 stride |

## 1.4 进阶知识

### 1.4.1 stride(步幅)— 形状操作的底层

PyTorch 的 tensor 不是简单的"几维数组",而是一个**数据指针 + 元数据**:

```
tensor = storage + (shape, stride, offset)
```

- `storage`:一块连续的 1 维内存
- `shape`:每维大小
- `stride`:每维跨多少个元素才到下一个索引
- `offset`:起点偏移

`view` 和 `reshape` 之所以能"零成本"改形状,就是因为它们只改 `shape` 和 `stride`,不动 `storage`。

例子:`torch.arange(12)` 的 storage 是 `[0, 1, 2, ..., 11]`,`reshape(3, 4)` 之后:
- `shape = (3, 4)`,`stride = (4, 1)`
- `x[1, 2]` 实际访问 storage 的 `1*4 + 2*1 = 6` 号位

转置后 `stride` 变成 `(1, 4)`,内存不连续 —— `view` 不行,`contiguous()` 才会真正重新排 storage。

### 1.4.2 contiguous vs non-contiguous 性能差

GPU 计算喜欢**连续访存**(`coalesced access`)。不连续的 tensor 在 GPU 上算卷积、矩阵乘法,可能比连续版本**慢 5-10 倍**。

判断:`tensor.is_contiguous()` 返回 `True` 才算连续。

什么时候会不连续?
- `transpose` / `permute` 之后
- 复杂的索引和切片

### 1.4.3 内存共享的隐藏陷阱

```python
x = torch.zeros(3, 3)
y = x[:, 0]            # y 跟 x 共享底层 storage
y += 1                  # in-place,会同时改 x
print(x[:, 0])          # tensor([1., 1., 1.])
```

`+=`、`*=`、`fill_()`、`zero_()` 这种 in-place 操作会**直接改原 tensor**。如果 y 是 loss 计算图里的一环,这种 in-place 操作可能会**让 autograd 报错**,因为 autograd 不喜欢看到输入被偷偷改了。

### 1.4.4 torch.contiguous_format vs torch.channels_last

PyTorch 2 套内存布局:
- `torch.contiguous_format`(默认):`(N, C, H, W)`,标准
- `torch.channels_last`:NHWC,GPU 上某些卷积算得更快(尤其 Tensor Core 优化)

切换:
```python
x = x.to(memory_format=torch.channels_last)
```

`torchvision` 训练时可以传 `memory_format=torch.channels_last` 给模型提升速度,这是 **PyTorch 2.0+ 的官方优化建议**。

## 1.5 学习自检(对应 14 条目标)

1. ✅ 能口述 Tensor / shape / dtype / device / requires_grad / CPU / GPU / CUDA / 显存 / 梯度
2. ✅ 用 `torch.tensor` / `ones` / `zeros` / `randn` 创建
3. ✅ 用 6 个属性检查 tensor
4. ✅ 用 `numel() * element_size()` 估算内存
5. ✅ `torch.cuda.is_available()` 判断 CUDA
6. ✅ `.to(device)` 迁移设备,理解返回新 tensor
7. ✅ `from_numpy` 共享 / `torch.tensor` 复制
8. ✅ `.cpu().numpy()` 必要性
9. ✅(02 详细讲)跑通 forward → loss → backward 看到 grad
10. ✅ 读懂 `memory_allocated` / `max_memory_allocated` / `empty_cache`
11. ✅ 标量 / 向量 / 矩阵 / 批次张量的 shape 读法
12. ✅ 二维张量取行/取列/取单值
13. ✅ 解释 `(4,4) / (4,)` 怎么广播
14. ✅ view / reshape / clone / contiguous 内存关系

## 1.6 常见坑

| 现象 | 原因 | 修法 |
|------|------|------|
| `RuntimeError: view size is not compatible` | 想 view 的 tensor 不连续 | 先 `.contiguous()` 再 view,或直接用 `.reshape()` |
| 改了一个变量,另一个跟着变 | view 共享内存 | 用 `.clone()` 切断 |
| `numpy()` 报 cuda 错 | GPU tensor 不能直接转 numpy | 先 `.cpu()` |
| 广播报错 `The size of tensor a must match...` | 不满足广播规则 | 检查两边 shape 对齐情况 |

## 1.7 下一步

进入 [`02-autograd/doc/02-autograd.md`](../../02-autograd/doc/02-autograd.md) 学自动求导。
