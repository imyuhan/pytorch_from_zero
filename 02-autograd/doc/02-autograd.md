# 02 - 自动求导 Autograd(教学文档)

> autograd 是 PyTorch 的灵魂,也是跟 NumPy **最大的区别**。学完你应该能口述 `requires_grad` / `grad` / `grad_fn` / `backward` / `no_grad` 五件套,能手写一次完整的"forward → loss → backward → update"循环。

## 2.1 涉及的文件

| 文件 | 主题 |
|------|------|
| `examples/01_requires_grad.py` | `requires_grad` + `grad` + `grad_fn` 动态追踪 |
| `examples/02_forward_backward.py` | 最小线性回归一步,看 grad_fn 链路 |
| `examples/03_grad_accumulate.py` | 梯度累积 + `zero_grad()` |
| `examples/04_no_grad.py` | `torch.no_grad()` 推理 + 显存对比 |
| `examples/05_manual_gd.py` | 手写梯度下降(带 LR) |

## 2.2 基础知识

### 2.2.1 什么是自动求导

**手算梯度**(以 `y = w*x + b`,求 `dL/dw` 为例):

```
正向: y = w*x + b       →   L = (y - y_true)^2
反向: dy/dw = x          →   dL/dw = 2*(y - y_true)*x
```

网络一大(几百万参数),这种链式推导靠人手写不可能。autograd 的核心思路:

1. 你**只写 forward**(正向)
2. PyTorch **自动**记住每一步用了什么算子
3. 一行 `.backward()`,PyTorch **自动**用链式法则反传所有梯度

这就是 **define-by-run**(动态图)的威力 —— 你的 Python 代码怎么跑,计算图就怎么长。

### 2.2.2 三件套:requires_grad / grad / grad_fn

| 属性 | 在哪 | 含义 |
|------|------|------|
| `tensor.requires_grad` | 所有 tensor | 是否要建计算图。**True → autograd 会记录所有 op** |
| `tensor.grad` | **叶子** tensor | 反向传播后存梯度值,初始为 `None` |
| `tensor.grad_fn` | **非叶子** tensor | 记录这个 tensor 是从哪个 op 来的,初始为 `None` |

**叶子 vs 非叶子**:
- **叶子**(leaf):用户**直接创建**的 tensor(`torch.tensor(...)`、`torch.randn(...)` 等)
- **非叶子**:由 op 产生(`y = x**2` 里的 `y`)

只有叶子节点的 `requires_grad=True` 时,backward 才会给它们累加 `.grad`。非叶子节点梯度**反传完就释放**,除非你 `retain_graph=True`。

### 2.2.3 一个最小的例子

```python
w = torch.tensor(2.0, requires_grad=True)
y = w ** 2             # y = w^2
y.backward()            # dy/dw = 2w = 4
print(w.grad)           # tensor(4.)
```

只有一行 `requires_grad=True` + 一行 `backward()`,梯度就出来了。这就是 autograd 的全部 API。

### 2.2.4 `.backward()` 的几个细节

```python
loss.backward()
```

- 只能对**标量**调用(`loss` 必须是 0 维的 `tensor`)
- 如果是**向量**(比如 loss 是按 batch 平均后的张量),要传 `gradient` 参数:

```python
y = model(x)              # shape [B, 10]
loss_per_sample = loss_fn(y, target)   # shape [B]
loss_per_sample.backward(gradient=torch.ones_like(loss_per_sample))
```

更常见的做法是**先 mean / sum**:
```python
loss = loss_fn(y, target).mean()  # 标量
loss.backward()
```

### 2.2.5 梯度累积与 `zero_grad()`

```python
for xb, yb in loader:
    loss = criterion(model(xb), yb)
    loss.backward()       # 梯度**累加**到 .grad(不是覆盖)
    optimizer.step()
    optimizer.zero_grad() # 清空 .grad,下个 batch 不受上批影响
```

**容易踩坑**:忘了 `zero_grad()` 会导致梯度**逐 batch 累加**,等效于把 batch_size 放大 2 倍、3 倍……,loss 表现得很奇怪。

**为什么默认累加而非覆盖?**
- 累加是**通用 API**:支持你手动攒多个 batch 的梯度再更新(gradient accumulation,显存不够时常用)
- `optimizer.zero_grad()` 帮你**在标准训练循环里**清零

### 2.2.6 `torch.no_grad()` 和 `detach()`

两者都让 tensor **脱离计算图**,但用起来有差异:

| 方法 | 形式 | 影响 |
|------|------|------|
| `with torch.no_grad():` | 上下文管理器 | **块内所有 op 都不建图**(推理时常用) |
| `tensor.detach()` | 方法 | 返回一个**新 tensor**,跟原 tensor 共享数据但**不在计算图里** |

```python
# 推理(必须 no_grad,否则显存爆)
model.eval()
with torch.no_grad():
    y_pred = model(x_test)        # 显存小、不建图

# 截断
y = model(x)
y_no_grad = y.detach()             # y_no_grad.requires_grad = False
y_no_grad.sum().backward()          # 反向不会传到 model
```

`no_grad` 还能**省显存** —— 因为不建图,PyTorch 不会保留激活值用于反传,显存立刻降下来。

## 2.3 逐个 example 讲解

### 2.3.1 `01_requires_grad.py` — 动态追踪

```python
x = torch.tensor(3.0, requires_grad=True)
print(f"x:        value={x.item()}, grad={x.grad}, grad_fn={x.grad_fn}")
# grad=None, grad_fn=None  ← 因为 x 是叶子
```

叶子节点的 `grad_fn` 永远是 `None` —— 它没有"来源",是自己凭空创建的。

```python
y = x ** 2
print(f"y:        value={y.item()}, grad_fn={y.grad_fn}")
# grad_fn=<PowBackward0>
```

`PowBackward0` 表示 `y` 是**幂运算**产生的。`backward` 算子名 = `Backward0`,前缀是**反向时**会跑的算子(所以 Pow 的反向是求 2x)。

```python
z = y * 2 + 1
print(f"z:        grad_fn={z.grad_fn}")
# grad_fn=<AddBackward0>
```

整个链路:

```
x (叶子, requires_grad=True)
 ↓ PowBackward
y = x^2
 ↓ MulBackward
z = 2y + 1
```

`z.backward()` 时,梯度沿箭头**反向**走链式法则,最终累加到叶子 `x.grad`。

```python
z.backward()
print(f"x.grad = {x.grad}")  # 12.0
```

数学验证:`z = 2x^2 + 1, dz/dx = 4x`,代入 `x=3` 得 `12`。✓

### 2.3.2 `02_forward_backward.py` — Forward / Backward 链路

```python
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])

y_pred = w * x + b
loss = ((y_pred - y_true) ** 2).mean()
```

这是个**最简线性回归**一步。链路上的 `grad_fn`:

```
w ─┐
   ├─→ MulBackward → y_pred ─→ SubBackward → err ─→ PowBackward → err^2 ─→ MeanBackward → loss
b ─┘
```

`loss` 的 `grad_fn` 是 `MeanBackward0` —— 因为最后一步是 `.mean()`。

```python
loss.backward()
print(f"  w.grad = {w.grad}")
print(f"  b.grad = {b.grad}")
```

`w.grad` 和 `b.grad` 是**标量梯度**,告诉优化器"参数该往哪边挪、挪多大"。

### 2.3.3 `03_grad_accumulate.py` — 梯度累加

```python
w = torch.tensor(2.0, requires_grad=True)
y1 = w ** 2
y1.backward()
print(w.grad)               # 4.0

y2 = w ** 3
y2.backward()               # 不清零,继续累加
print(w.grad)               # 16.0
```

数学上 `dy1/dw + dy2/dw = 2*2 + 3*4 = 16`,`y1` 和 `y2` 都反向,所以 `w.grad` 是两次的**累加**。

```python
w.grad.zero_()
y3 = w ** 2
y3.backward()
print(w.grad)               # 4.0  ← 干净的值
```

`tensor.zero_()` 是 in-place 把所有元素清零。下划线后缀 = in-place 版本。

### 2.3.4 `04_no_grad.py` — 推理模式

```python
w = torch.randn(3, 3, requires_grad=True)
y = (w * 2).sum()
print(y.requires_grad)        # True
print(y.grad_fn)              # <SumBackward0>

with torch.no_grad():
    y_inf = (w * 2).sum()
    print(y_inf.requires_grad)  # False
    print(y_inf.grad_fn)        # None
```

在 `no_grad` 上下文里,**PyTorch 不会记录任何 op**,出来的 tensor 是普通 tensor,`requires_grad=False`、`grad_fn=None`。

```python
y_det = (w * 2).sum().detach()
print(y_det.requires_grad)    # False
```

`.detach()` 也行,效果一样:返回**与原 tensor 共享数据**、**不在计算图里**的新 tensor。

#### 显存对比

```python
big = torch.randn(1000, 1000, device="cuda", requires_grad=True)  # 4MB 浮点
# 但 PyTorch 还要为它保留计算图的开销,实际占 ~8MB
mem_with_grad = torch.cuda.memory_allocated() / 1024**2  # 3.81 MB(只算 tensor 本身)
```

**`no_grad` 内的优势**:
- 不存激活 → 显存立刻降
- 不建计算图 → 算得略快

**实战**:eval loop 一定要包 `no_grad`,否则:
- 显存累积,跑着跑着 OOM
- 算力浪费在不需要的图构建上

### 2.3.5 `05_manual_gd.py` — 手写梯度下降

目标函数:`f(w) = (w - 5)^2`,理论最优 `w* = 5`,梯度 `df/dw = 2(w - 5)`。

```python
w = torch.tensor(0.0, requires_grad=True)
lr = 0.1

for step in range(8):
    loss = (w - 5) ** 2
    loss.backward()

    with torch.no_grad():           # 关键:更新时不建图
        w -= lr * w.grad

    w.grad.zero_()
```

**三步**:
1. `loss.backward()`:求梯度,累加到 `w.grad`
2. `w -= lr * w.grad`:用梯度**更新参数**(`-=` 是 in-place)
3. `w.grad.zero_()`:清零,下个 step 干净

**为什么用 `with no_grad`?** 更新参数本身**不需要**被 autograd 追踪(否则 autograd 会想"咦,这个变量自己也变了,要不要算它的梯度?"——会出问题)。

跑完 8 步,`w` 从 0 → 4.16,接近理论值 5。继续跑下去会指数逼近。

**真正的 optimizer**(`torch.optim.SGD`、`torch.optim.Adam`)内部就是干这个,**只是更花哨**(动量、自适应学习率等)。

## 2.4 进阶知识

### 2.4.1 计算图的可视化

```python
import torch
from torchviz import make_dot

x = torch.tensor(1.0, requires_grad=True)
y = x ** 2
z = y * 3 + 1
make_dot(z, params={"x": x})  # 生成计算图
```

`torchviz` 会画出节点(算子)和边(tensor),debug 复杂模型时救命。

### 2.4.2 雅可比-向量积 (Jacobian-Vector Product)

`.backward()` 实际算的不是 Jacobian 矩阵本身,而是 **Jacobian × 一个向量**(链式法则的核心)。

```python
x = torch.randn(3, requires_grad=True)
y = x ** 2                    # y 是向量 [3]
# 想 dL/dx,但 L 怎么定义?
y.backward(gradient=torch.ones(3))  # 显式指定"上游梯度"全为 1
print(x.grad)                  # 等价于对 y.sum() 反向 → 2x
```

这就是为什么 `loss = criterion(...).mean()` 必须先 reduce 成标量,或显式传 `gradient` 参数 —— `backward` 算的是 **JVP**(雅可比-向量积),不是 Jacobian。

### 2.4.3 高阶梯度(高阶导数)

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3                     # y = x^3
g = torch.autograd.grad(y, x, create_graph=True)[0]  # g = 3x^2 = 12
g2 = torch.autograd.grad(g, x)[0]                     # g2 = 6x = 12
```

`create_graph=True` 让反向时**也建图**,这样可以再反向一次。用于:
- 物理仿真(力的梯度 → 加速度)
- 元学习(MAML)
- 优化器的优化(是的,可以对优化器求导)

### 2.4.4 PyTorch 动态图 vs TensorFlow 静态图

| | PyTorch(动态) | TensorFlow 1.x(静态) |
|---|---|---|
| 图构建时机 | **每次前向重新建** | 一次定义,反复跑 |
| 调试 | 跟普通 Python 一样 print/pdb | 需要特殊工具 |
| 条件分支 | 自由 if/while | 难(需要 tf.cond) |
| 部署 / 优化 | 早期弱(已改善) | 强(TF Lite 等) |

PyTorch 2.x 的 `torch.compile()` 把动态图编译成静态图,保留易用性 + 拿到静态图的性能。

### 2.4.5 内存优化:`set_grad_enabled`

```python
torch.set_grad_enabled(False)   # 全局关闭
# ... 推理代码 ...
torch.set_grad_enabled(True)    # 再打开
```

等价于手动包一层全局 `no_grad`。在 eval loop 入口、或者反复在 train/eval 切换的脚本里有用。

### 2.4.6 `retain_graph` 参数

默认情况下,反向传播完计算图就**释放**了,二次 `.backward()` 会报错:

```python
loss.backward()
loss.backward()  # 报错:RuntimeError: Trying to backward through the graph a second time
```

如果要二次反向(例如 GAN 的两个 loss 共享计算图):
```python
loss1.backward(retain_graph=True)
loss2.backward()
```

**实战很少用**。大多数情况一个 loss 算一次反向即可。

## 2.5 常见坑

| 现象 | 原因 | 修法 |
|------|------|------|
| `RuntimeError: grad can be implicitly created only for scalar outputs` | 试图对向量 backward | 先 `.mean()` / `.sum()`,或传 `gradient` |
| `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation` | in-place 改了叶子 tensor | 用 `w.data -= lr * w.grad` 或用 optimizer |
| 训练时 loss 震荡 / 越训越差 | 忘记 `zero_grad()` | 每 step 加 `optimizer.zero_grad()` |
| eval 时显存爆 | 推理也建图 | 包裹 `with torch.no_grad():` |
| `tensor.grad` 是 `None` | 1) 非叶子;2) `requires_grad=False`;3) 没反向过 | 检查三件 |

## 2.6 学习自检

能口述:
- [ ] `requires_grad=True` 打开后会发生什么?
- [ ] `backward()` 为什么只能对**标量**调用?
- [ ] 梯度是**累加**还是**覆盖**?为什么?
- [ ] `no_grad` 和 `detach` 的区别?
- [ ] 训练循环 4 行标准结构?

能写:
- [ ] 一个手写 GD 收敛到目标值
- [ ] 修一个"`backward` 二次报错"的问题
- [ ] 修一个"`zero_grad` 忘加"的 bug

## 2.7 下一步

进入 [`03-dataset-dataloader/doc/03-dataset-dataloader.md`](../../03-dataset-dataloader/doc/03-dataset-dataloader.md) 学数据加载。
