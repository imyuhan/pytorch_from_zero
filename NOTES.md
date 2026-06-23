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

## 02-autograd / 浮点限制:`requires_grad` 只能用 float / complex

> 跟 `## 04-attention` 平行的章节(都属大章节,但 autograd 是第二章、attention 是第四章)。这里是 autograd 入门第一个坑。

PyTorch 的 `requires_grad=True` **只接受浮点( float32 / float64 / float16 )和复数( complex64 / complex128 )类型**,整数 / 布尔会直接报错:

```
RuntimeError: Only Tensors of floating point and complex dtype can require gradients
```

### 报错典型场景

```python
import torch

w = torch.tensor(2, requires_grad=True)
# RuntimeError:Only Tensors of floating point and complex dtype can require gradients
```

`torch.tensor(2)` 默认推断成 `torch.int64`(Long),整数没有"瞬时变化率",所以 autograd 拒绝。

### 修法(按推荐度排)

```python
# 1. 加小数点,最简单
w = torch.tensor(2.0, requires_grad=True)                     # ✅ float64

# 2. 显式指定 dtype
w = torch.tensor(2, dtype=torch.float32, requires_grad=True) # ✅

# 3. 用工厂函数(默认就是 float,根本不会撞)
w = torch.randn(1, requires_grad=True)                       # ✅ float32
w = torch.zeros(3, requires_grad=True)                       # ✅ float32
```

### 为什么有这个限制

**"梯度"是连续数学的概念**,对实数 / 复数才有定义;整数 / 布尔没有"稍微变一点点"的连续变化,所以 PyTorch 在源头就禁止了。

### 哪些 dtype 能 `requires_grad`,哪些不能

| dtype | 能 `requires_grad` 吗 | 用途 |
|---|---|---|
| `torch.float32`(默认)/ `float64` / `float16` | ✅ | 模型权重、loss、输入数据 |
| `torch.complex64` / `complex128` | ✅ | 复数信号处理(罕见) |
| `torch.int32` / `int64`(默认整数) | ❌ | 标签 index、类别编号 |
| `torch.bool` | ❌ | mask、布尔索引 |

### 易混点:标签本来就不该 requires_grad

```python
y_true = torch.tensor([0, 1, 2])                          # int64 标签,正常,不需要 grad
y_pred = torch.randn(3, 3, requires_grad=True)            # float 预测,需要 grad
loss = nn.CrossEntropyLoss()(y_pred, y_true)              # OK!标签不用 requires_grad
```

标签是真实答案,本来就不是模型要"学"的参数——而且它本来就是整数,也没法 `requires_grad`。

### 一句话总结

> **`requires_grad=True` 只能用在浮点张量上**。`torch.tensor(2)` 改成 `torch.tensor(2.0)` 或加 `dtype=torch.float32` 即可。

## 02-autograd / f-string 打印带 grad 的 tensor 会悄悄清掉 `.grad`

在 `02-autograd/exercises/04_linear_step.py` 里踩到的隐藏小坑:

```python
print(f"w: {w:>.4f}, w.grad: {w.grad:>.4f}, ...")   # 这一行会清掉 w.grad / b.grad
with torch.no_grad():
    w -= lr * w.grad                                # 这里 w.grad 已经是 None,直接报错
```

### 为什么会清掉

f-string 里的 `{w:>.4f}` 触发链路:
1. Python 调 `format(w, '.4f')`
2. Tensor 没实现 `__format__`,退回去调 `str(w)` → 走 `__repr__`
3. **PyTorch 的 leaf tensor 在 `__repr__` 时会顺带检测 grad 状态,某些路径下会把 `.grad` 置成 `None`**(并打 `UserWarning: None of the inputs have requires_grad=True. Gradients will be None` 之类的告警)

更隐蔽的是:**就算当场不报 warning,下次 backward() 也会出问题** —— 因为 `.grad` 已经被清,`with torch.no_grad(): w -= lr * w.grad` 拿到的是 `None`。

### 修法(按推荐度排)

1. **先 `.item()` 再格式化(推荐)**:`{w.item():.4f}`、`{w.grad.item():.4f}`
   - 优势:只读标量、纯 Python float,完全不动 grad 状态
2. **`{w.detach()}`**:同样能拿到不带梯度的值,但返回的还是 tensor
3. **临时 `.clone()`**:笨但能跑,`{w.clone():.4f}`,本质同 `.item()`

### 顺手养成的好习惯

真实训练循环里**每个 step 打印 loss / metric 之前,先 `.item()` 一下**,除了避开这个小坑,还有两个隐性收益:
- **少遍历一遍计算图**:`__repr__` 会走整张计算图,大模型里每次 print 都能感觉到卡顿
- **避免 print 触发的副作用**:清 grad / 重复 backward / 修改 inplace 这些"看起来只读"的坑,基本都是 `__repr__` / `__str__` 埋的

### 一句话总结

> **别在 f-string 里直接放带 grad 的 tensor**。要打印值就先 `.item()` —— 一字之差,避开一个隐藏雷。

## 02-autograd / autograd 三件套:`requires_grad` / `grad_fn` / `grad`

autograd 里有三个名字像、含义完全不同的概念,集中对比,以后看报错不再懵。

- **`requires_grad`** —— 张量的**静态属性**(bool),声明"我愿不愿意被追踪、要不要参与梯度计算"
- **`grad_fn`** —— 张量的**"出身证明 + 反向函数指针"**,指向生成它的那次运算对应的反向函数(`<PowBackward0>` 这种)
- **`grad`** —— 张量上的**动态数据**(Tensor 或 None),装的是别人反向传播算完传过来的梯度值

类比:`requires_grad` 是"我愿不愿意被拍照"的开关,`grad_fn` 是"我是被哪台相机拍的"(决定反传时该找谁),`grad` 是"照片本身"。

### 对比表

| 维度 | `requires_grad` | `grad_fn` | `grad` |
|---|---|---|---|
| **类型** | `bool`(True / False) | 反向函数对象 或 `None` | `Tensor` 或 `None` |
| **什么时候定** | **创建张量时**就定下来了 | 张量**被算出来时**就挂上了 | 反向传播**之后**才有 |
| **控制什么** | 这个张量是否要建计算图、追踪梯度 | 反向传播时该调哪个函数算局部梯度 | 这个叶子节点收到的梯度值 |
| **谁设置它** | 你(`requires_grad=True`)或上游传播 | autograd(自动,根据产生它的运算) | autograd(`.backward()` 时自动填) |
| **存不存在** | 永远存在(默认 False) | 叶子恒为 `None`;被追踪的中间节点都有 | 没反向传播过就是 `None` |
| **修改方式** | `x.requires_grad_(True)`(原地) | 不能直接改(只能切断:`detach()`) | `x.grad.zero_()`、`x.grad += ...` |

### 直观跑一下(三个概念一张图都看到)

```python
import torch

w = torch.tensor(3.0, requires_grad=True)
print("=== 叶子 w ===")
print("w.requires_grad:", w.requires_grad)   # True
print("w.grad_fn:", w.grad_fn)               # None(叶子,没有"上一步")
print("w.grad:", w.grad)                     # None(没反向传播)

y = w ** 2
print("\n=== y = w**2 ===")
print("y.requires_grad:", y.requires_grad)   # True(传染)
print("y.grad_fn:", y.grad_fn)               # <PowBackward0>(幂运算的反向函数)
print("y.grad:", y.grad)                     # None(不是叶子,不算)

z = y.sum()
print("\n=== z = y.sum() ===")
print("z.requires_grad:", z.requires_grad)   # True
print("z.grad_fn:", z.grad_fn)               # <SumBackward0>
print("z.grad:", z.grad)                     # None

z.backward()                                  # 反向传播开始算
print("\nw.grad:", w.grad)                    # tensor(6.) = 2*w
```

### 三个易混点

#### 1. `requires_grad` 是会"传染"的

```python
w = torch.tensor(3.0, requires_grad=True)
y = w * 2
z = y * 3
print(z.requires_grad)  # True,w 开过门,后面整条链都愿意被追踪
```

这就是为啥 `no_grad` / `detach` 重要 —— **从某个点开始"关门"**,后面的就不追踪了。

#### 2. 叶子节点 `grad_fn` 一定是 `None`,因为没有"上一步"

```python
w = torch.tensor(3.0, requires_grad=True)
w.grad_fn  # None
```

判断"是不是叶子"看 `is_leaf`;叶子 + 浮点 + `requires_grad=True` 才有资格存 `grad`。

#### 3. `no_grad` / `detach` 出来的张量:`requires_grad=False` + `grad_fn=None` + `is_leaf=False` 三连

```python
with torch.no_grad():
    a = w ** 2
print(a.requires_grad, a.grad_fn, a.is_leaf)   # False None False

y3 = (w ** 2).sum().detach()
print(y3.requires_grad, y3.grad_fn)            # False None
```

这"三连"几乎一定意味着是 no_grad / detach 出来的 —— 反向传不回去。

### 报错一次性看明白

#### 报错 1:`w.grad` 是 `None`

```
AttributeError: 'NoneType' object has no attribute 'zero_'
```

```python
w = torch.tensor(3.0, requires_grad=True)   # 门开对了
y = (w ** 2).sum()                            # 算出来了,但没调 backward
w.grad.zero_()                                # 炸!grad 还是 None
```

**`requires_grad=True` 门开对了,但没触发 `backward()`,`grad` 容器里就没东西。**

#### 报错 2:`y2.backward()` 失败

```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

```python
with torch.no_grad():
    y2 = w ** 2
y2.backward()   # 炸
```

PyTorch 一次说了两件事:`requires_grad=False` + `grad_fn` 是 `None`,**两者是同一回事** —— 没追踪就一定没 grad_fn,backward 无从下手。

### 反向传播到底在干什么

`z.backward()` 触发后,PyTorch 从 `z` 出发,**沿着 `grad_fn` 链逆推**:

```
z (grad_fn=SumBackward0)  ←── z.backward() 从这里出发
       ↑ 调 SumBackward0
y (grad_fn=PowBackward0)
       ↑ 调 PowBackward0
w (grad_fn=None, 叶子)     ←── 链到此为止,把累积梯度写到 w.grad
```

只要中间任何一环 `grad_fn` 是 `None`(比如 `no_grad` 里的张量),**链就断了,backward 报错**。

### 一张图全串起来

```
创建 w (requires_grad=True)    ←── 开关打开
       │
       ▼ w**2
y (grad_fn=PowBackward0)        ←── 挂上"出身证明"
       │
       ▼ y.sum()
z (grad_fn=SumBackward0)
       │
       ▼ z.backward()
   沿 grad_fn 链逆推 ──► w
                            │
                            ▼
                       w.grad = 6.0
```

### 一句话总结

> - `requires_grad` —— **开关**(愿不愿意被追踪)
> - `grad_fn` —— **出身证明 / 反向函数指针**(反传时该调谁)
> - `grad` —— **结果**(叶子收到的梯度值)
>
> `no_grad` / `detach` 切断的是"开关"和"出身证明",所以 `backward` 走不下去、`grad` 也填不上。三件套一损俱损。

## 04-attention / Attention 4 个关键知识点

学 Attention 时绕不开的几个底层机制,集中整理一下。

### 1. 为什么除以 `sqrt(d_k)` 而不是别的值

Transformer 的 Scaled Dot-Product Attention 里有个
`Q @ K.T / sqrt(d_k)`,这个除法不是经验值,是**数学上严格推导**出来的。

**问题**:Q、K 的元素假设均值为 0、方差为 1(初始化或 LayerNorm 保证),
那么它们点积 `Q·K` 的方差会等于维度 d_k——**维度越大,点积数值越大**。

**后果**:点积直接喂给 softmax,数值太大 → softmax 进入饱和区 →
概率分布接近 one-hot → **梯度趋近于 0**,训练崩。

**解决**:除以 `sqrt(d_k)`。

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

为啥是 sqrt(d) 而不是别的?

| 选择 | 效果 |
|---|---|
| 除以 d | 方差变成 1/d,数值压得太小,softmax 输出太平 |
| 除以 sqrt(d) | 方差刚好回到 1,**数学上最自然的选择** |
| 不除 | 维度一大就梯度消失 |
| 乘个可学习系数 | 也行(部分模型如 Linformer 这么做),但默认这个最稳 |

**一句话**:除以 `sqrt(d_k)` 是为了让 `Q·K` 的方差回到 1,防止 softmax 饱和导致梯度消失。

---

### 2. Softmax

把一组任意实数 → 一组概率分布(和为 1,每个值在 0~1)。

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

```python
import torch
import torch.nn.functional as F

x = torch.tensor([1.0, 2.0, 3.0])
F.softmax(x, dim=0)
# tensor([0.0900, 0.2447, 0.6652])  ← 和=1,最大值占大头
```

**核心作用**:放大差异 + 归一化成概率。

#### 参数

| 参数 | 作用 |
|---|---|
| `input` | 输入张量,任意形状 |
| `dim` | **沿哪个维度归一化**(最关键) |
| `dtype` | 输出类型,防 `exp()` 数值溢出 |

**`dim` 怎么算**:`dim=k` 表示沿第 k 维归一化,**这一维算完和为 1**。
负数 `-1` 等价于最后一维,`-2` 等价于倒数第二维,公式是
`dim=-k` ↔ `dim=总维数-k`。

```python
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

F.softmax(x, dim=0)  # 沿行归一化 → 每列和=1
F.softmax(x, dim=1)  # 沿列归一化 → 每行和=1
F.softmax(x, dim=-1) # == dim=1,最后一维
```

**Attention 里**:scores shape = (batch, heads, seq_q, seq_k),
`dim=-1` 让**每行(每个 query 对所有 key)**和为 1,得到注意力权重。

#### 防溢出技巧

```python
# 大数会爆
F.softmax(torch.tensor([1000.0, 1001.0, 1002.0]), dim=0)
# tensor([nan, nan, nan])  ← exp(1000) 已经溢出

# 解决:用 float32 算,再转回去
F.softmax(x, dim=-1, dtype=torch.float32).to(x.dtype)
```

---

### 3. One-Hot(独热编码)

**类别是离散的标签**,神经网络只认数字,所以要用一串 0/1 表示一个类别。
对应位置是 1,其他全是 0——**只有一个位置是"热的"**,所以叫 one-hot。

| 类别 | One-hot |
|---|---|
| 猫 | `[1, 0, 0, 0]` |
| 狗 | `[0, 1, 0, 0]` |
| 鸟 | `[0, 0, 1, 0]` |
| 鱼 | `[0, 0, 0, 1]` |

```python
F.one_hot(torch.tensor([0, 1, 2, 3]), num_classes=4)
# tensor([[1, 0, 0, 0],
#         [0, 1, 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1]])
```

#### 跟 softmax 的关系

- **softmax 输出**:连续概率,比如 `[0.05, 0.85, 0.10]`
- **one-hot**:硬标签,比如 `[0, 1, 0]`

训练目标就是让 softmax 输出**尽量接近** one-hot,这一步由
**交叉熵 Loss** 完成(softmax 输出 → log → 跟 one-hot 对位相乘再取负)。

---

### 4. `scores.masked_fill(~key_mask, float("-inf"))`

Attention 里的 mask 操作,**让某些位置"看不见"**(通常用来屏蔽 padding)。

#### 一行行拆

```python
scores = scores.masked_fill(~key_mask, float("-inf"))
```

| 部分 | 含义 |
|---|---|
| `scores` | Q·K 算出的分数矩阵 (batch, heads, seq_q, seq_k) |
| `masked_fill` | 按 mask 填值,True 的位置填,False 的位置保留 |
| `~key_mask` | `~` 是按位取反,**翻转 True/False** |
| `float("-inf")` | 负无穷,Python 里就是 `-math.inf` |

#### key_mask 的语义

通常 `key_mask` 里 **True = 有效位置,False = padding**。
取反后 `~key_mask` 就是 **padding 位置变 True**,
`masked_fill` 把这些位置填成 `-inf`。

#### 为啥是 `-inf` 而不是 0?

因为下一步要进 softmax。softmax = `e^x / Σe^x`:

| 填充值 | softmax 后 | 效果 |
|---|---|---|
| `-inf` | 0 | ✅ padding 完全不参与注意力 |
| `0` | ≈ 0.25(均分) | ❌ padding 还在抢权重 |
| `-1e9` | ≈ 0 | ✅ 数值上也行,但 `-inf` 最干净 |

**数学原理**:`e^(-inf) = 0`,所以 padding 位置权重恰好是 0。
任何有限数都做不到这一点。

#### 完整 Attention 链路

```python
# 1. Q·K 算原始分数
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)

# 2. mask:把 padding 设成 -inf(key_mask True=有效)
scores = scores.masked_fill(~key_mask, float("-inf"))

# 3. softmax → 注意力权重(padding 位置自动是 0)
attn = F.softmax(scores, dim=-1)

# 4. 加权求和
output = attn @ V
```

**一句话**:`masked_fill(~key_mask, -inf)` = "告诉模型,padding 位置你别看",靠 softmax 的数学性质让那里权重变 0。
