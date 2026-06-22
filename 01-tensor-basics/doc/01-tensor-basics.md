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

### 1.4.5 深度学习核心概念速览

这一节是术语速查表,每条把概念本身讲清楚。学 DL 时随时翻——后面章节展开时这里只做回顾。

> 命名约定:小标题里的"术语(英文)"格式只是为了查表时好搜,文中叙述只用中文。

---

#### 1) 模型(Model)

把输入映射到输出的**可学习函数**。结构(几层、用什么激活)由人写,内部参数由数据训出来。

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)   # 一层线性变换:输入 784 维 → 输出 10 维
    def forward(self, x):
        return self.linear(x)

model = MyModel()
y = model(x)        # 触发 forward,得到预测
```

PyTorch 里模型就是一个继承 `nn.Module` 的类,`model(x)` 是入口。

---

#### 2) 参数 / 权重(Parameters / Weights)

模型里**会被训练调整的数字**,通常是矩阵或向量。训练的本质 = 找一组让 loss 最小的参数。

```python
m = MyModel()
for p in m.parameters():
    print(p.shape)
```

参数数量怎么算:`nn.Linear(in, out)` 含 `in*out + out` 个(权重 + 偏置)。`nn.Linear(784, 10)` 就是 `784*10 + 10 = 7850`。

注意:`nn.Parameter` 是 `Tensor` 的子类,会自动加入 `model.parameters()`。普通 `torch.tensor` 不会,要手动包。

---

#### 3) 前向传播(Forward Pass)

输入顺着模型**算一遍**得到预测值。也叫"推理",只是训练时推理结果还要拿去算 loss。

```python
x = torch.randn(32, 784)   # batch_size=32
logits = model(x)          # (32, 10)
```

`model(x)` 等价于调用 `__call__`,按 `forward` 定义的顺序计算。前向传播**不学习**,参数没动——训练是前向 + 反向 + 更新三步走。

---

#### 4) 损失函数(Loss Function)

衡量**预测值和真实值的差距**,输出一个**标量**(越小越好)。把"模型答得对不对"翻译成一个数,优化器才好调参。

```python
criterion = nn.CrossEntropyLoss()    # 多分类(接受 logits + 类别索引)
# criterion = nn.MSELoss()         # 回归
# criterion = nn.BCEWithLogitsLoss()  # 二分类

logits = model(x)                   # (B, num_classes),未归一化的分数
target = torch.tensor([3, 7, 1])    # (B,),每个样本的真实类别索引
loss = criterion(logits, target)    # 标量
```

要点:loss 必须是**标量**才能 `.backward()`。

---

#### 5) 反向传播(Back Propagation, Backprop)

从 loss 出发,**用链式法则**算出每个参数对 loss 的影响(梯度)。一反向扫,效率极高。

```python
loss = criterion(model(x), target)
loss.backward()   # 计算图里所有叶子节点的 .grad 被填上
```

为什么叫"反向":梯度从 loss 往参数方向传,数学上等价于链式法则的**反向模式自动微分**——对 N 个参数、M=1 个 loss 的场景,反向模式只需一次扫;正向模式要 N 次扫。

---

#### 6) 梯度(Gradient)

损失对参数的**偏导数**。
- **方向**:负梯度方向 = loss 下降最快的方向
- **大小**:在那个方向上"动多少合适"的参考

参数 tensor 上的 `.grad` 属性就是它。

```python
for p in model.parameters():
    print(p.grad)        # 反向传播后被填上,形状跟参数一样
```

要点:`tensor.grad` 是属性不是方法,没有括号。初始是 None,backward 后是 Tensor。

---

#### 7) 优化器(Optimizer)

**根据梯度更新参数**的算法。最朴素的形式是 `θ ← θ - lr·grad`(SGD);更聪明的(如 Adam)会自适应调节每个参数的学习率。

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练循环三步口诀
optimizer.zero_grad()   # 1. 清空上轮梯度
loss = criterion(model(x), target)
loss.backward()         # 2. 算梯度
optimizer.step()        # 3. 按梯度更新参数
```

常见选择:
| 优化器 | 特点 |
|--------|------|
| SGD | 简单、可解释,CV 经典 |
| SGD + momentum | 加惯性,冲过小坑 |
| Adam | 自适应 lr,NLP / Transformer 首选 |
| AdamW | Adam + 解耦权重衰减,大模型标配 |

---

#### 8) 学习率(Learning Rate, lr)

`optimizer.step()` 一次迈多大。**最重要的超参数**。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

经验起点:
- SGD: 0.01 ~ 0.1
- Adam / AdamW: 1e-4 ~ 1e-3

直觉:太大 → 震荡不收敛;太小 → 训练龟速。

---

#### 9) Epoch / Batch / Iteration

数据被切碎喂给模型的三个粒度单位。

| 单位 | 含义 |
|------|------|
| **Batch Size** | 一块多少样本 |
| **Iteration** | 跑过几个 batch |
| **Epoch** | 全部样本过一遍 |

直觉:一本书 = 训练集。Batch = 一次读 10 页;Iteration = 读完 10 页这一动作;Epoch = 把书从头到尾读完一遍。

例:10000 样本 + `batch_size=100` → 1 epoch = 100 iteration;10 epoch = 1000 iteration。

```python
loader = DataLoader(dataset, batch_size=100, shuffle=True)
for epoch in range(10):          # 10 个 epoch
    for x, y in loader:          # 每个 epoch 内 100 个 iteration
        ...
```

---

#### 10) 训练集 / 验证集 / 测试集(Train / Val / Test)

数据三件套,**绝不能混**。

| 集合 | 用途 | 模型能不能看到 |
|------|------|------------|
| 训练集(Train) | 训参数 | ✅ |
| 验证集(Val) | 调超参、早停 | ❌ 只能评估 |
| 测试集(Test) | 最终汇报 | ❌ 只能用一次 |

```python
total = torch.utils.data.TensorDataset(X, y)
n = len(total)
train_set, val_set, test_set = torch.utils.data.random_split(
    total, [int(n*0.7), int(n*0.15), int(n*0.15)]
)
```

纪律:验证集只能用来调超参,测试集只能用一次——除非你确定要的就是这个最终成绩,否则别回头再基于 test 调模型。

---

#### 11) 过拟合 / 欠拟合(Overfitting / Underfitting)

模型"学得太死"或"学得太浅"两种病。

| 现象 | 训练集 | 验证集 |
|------|:------:|:------:|
| 欠拟合 | 差 | 差 |
| 刚好 | 好 | 好 |
| 过拟合 | 很好 | 变差 |

直观对比:

```
loss
 ↑
 │   训练loss        验证loss
 │     \              /
 │      \            / ← 验证loss反弹 = 过拟合起点
 │       \          /
 │        \________/
 └──────────────────────→ epoch
```

对抗手段一览:

| 现象 | 常用办法 |
|------|---------|
| 欠拟合 | 加容量(更宽/更深的模型)、训久点、换更强架构 |
| 过拟合 | 加数据、数据增强、Dropout、权重衰减、早停、减小模型 |

```python
# Dropout(训练时随机丢一半神经元,推理时关闭)
self.dropout = nn.Dropout(p=0.5)

# 权重衰减 / L2 正则(惩罚大权重,λ 越大越压权重)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
```

---

#### 12) 反向传播依赖的"基础三件套"——`requires_grad` / 计算图 / 反向模式自动微分

`loss.backward()` 一行求所有参数的梯度,背后是这三条在合作。

##### 12.1 `requires_grad` —— autograd 的开关

tensor 上的布尔标记。打开 = 这个 tensor 的所有运算会被 PyTorch"录音"。

```python
x = torch.randn(3, requires_grad=True)
y = x * 2
y.requires_grad   # True  ← 自动传播

# 模型参数默认就是 True
# 临时关闭(推理 / 算指标时省内存)
with torch.no_grad():
    y = model(x)   # 不建图
```

##### 12.2 计算图(Computational Graph) —— 录音的产物

前向时偷偷建的一张**有向无环图(DAG)**。节点是 tensor,边是运算(`+` `@` `ReLU` 等)。

- 为什么"有向":运算有顺序
- 为什么"无环":没有循环,否则图永远建不完

```python
a = torch.tensor(2.0, requires_grad=True)
b = a * 3        # 中间节点
c = b ** 2       # 中间节点
loss = c.sum()
# 这时图: a → b → c → loss
loss.backward()
print(a.grad)    # tensor(36.)  ← dloss/da = 2·3·3·a = 36
```

只有**叶子节点**(用户创建、`requires_grad=True` 的)有 `.grad`,中间节点没有。图在 `backward()` 后被释放(除非 `retain_graph=True`)。

##### 12.3 反向模式自动微分(Reverse-mode Autodiff)

从 loss 出发**反向**遍历计算图,用链式法则逐节点求梯度。

为什么选"反向":DL 模型常有上亿参数、但输出只有 1 个 loss。反向模式在这种"M=1 ≪ N"场景下,只需一次反向扫就能拿到所有 N 个梯度,计算量 ≈ 一次前向;正向模式要 N 次扫。

PyTorch 里这步对用户透明——`loss.backward()` 内部就是反向模式。

##### 三件套怎么联动

```
requires_grad=True
        ↓ 打开麦克风
前向传播(model(x))
        ↓ 边算边建图
计算图(DAG)
        ↓ loss.backward() 触发
反向模式自动微分
        ↓ 沿图反向遍历 + 链式法则
每个叶子节点的 .grad
        ↓ optimizer.step()
参数更新
```

具体推导、`backward()` 报错排查、`detach()` vs `no_grad` 的区别 —— 全在 02 章展开。

---

#### 一张图把 12 条串起来(训练循环全景)

```
┌─────────────────────────────────────────────────────┐
│  数据准备                                           │
│  训练集 → DataLoader 切 batch                       │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│  训练模式: model.train()                            │
│                                                     │
│  for epoch in range(num_epochs):                    │
│      for x, target in loader:                       │
│                                                     │
│          ① optimizer.zero_grad()  ← 清梯度          │
│          ② pred = model(x)         ← 前向传播       │
│          ③ loss = criterion(pred, target)  ← 损失   │
│          ④ loss.backward()          ← 反向传播       │
│          ⑤ optimizer.step()         ← 更新参数       │
│                                                     │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│  验证模式: model.eval()                             │
│      with torch.no_grad():  ← 省内存                │
│          val_loss / val_acc                         │
└─────────────────────────────────────────────────────┘
                      ↓
              最终:测试集评估一次
```

具体实现在 06-cnn/04_train_loop.py 和 07-experiments/02_train_mnist.py,先有个全局观。

---

#### 速查表

| # | 概念 | 一句话 | PyTorch 关键符号 |
|:-:|------|--------|----------------|
| 1 | 模型 | 输入→输出的可学习函数 | `nn.Module` |
| 2 | 参数 | 模型里会被训练的数字 | `model.parameters()` |
| 3 | 前向传播 | 顺着模型算一遍 | `model(x)` |
| 4 | 损失函数 | 衡量预测和真实的差距 | `nn.CrossEntropyLoss()` |
| 5 | 反向传播 | 从 loss 反算梯度 | `loss.backward()` |
| 6 | 梯度 | loss 对参数的偏导 | `param.grad` |
| 7 | 优化器 | 按梯度更新参数 | `torch.optim.Adam(...)` |
| 8 | 学习率 | 一步迈多大 | `lr=1e-3` |
| 9 | Epoch/Batch/Iter | 数据量单位 | `DataLoader(batch_size=N)` |
| 10 | 训练/验证/测试集 | 数据三件套 | `random_split(...)` |
| 11 | 过拟合/欠拟合 | 学太死 / 学太浅 | `Dropout`, `weight_decay` |
| 12 | requires_grad/计算图/反向autodiff | 反向传播的三件套 | `requires_grad=True`, `loss.backward()` |

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
