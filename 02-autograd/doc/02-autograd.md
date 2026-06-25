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

### 2.2.0 前置:求导、偏导、梯度、链式法则(autograd 的数学底子)

> 这一节是 autograd 的数学铺垫。如果你学过微积分可以快速过;没学过的话认真读完,看完会对"梯度"和"反向传播"有直觉。

#### 1. 为什么要求导?——训练神经网络的本质

神经网络有**一大堆参数**(几百万个权重),我们想让它"预测准"。

调参思路:
- 当前 loss = 5.0,想让它变小
- 怎么调?——**看 loss 对每个参数的"敏感度"**:敏感度高 → 大幅调;敏感度低 → 小幅调
- 这个"敏感度"就是**导数**

**一句话**:求导 = 算"参数稍微动一下,loss 会变多少、变好还是变坏"。

#### 2. 导数(derivative)

**定义**:函数 $f(x)$ 在 $x$ 处的导数 = $f$ 在该点的**瞬时变化率**。

几何意义:函数曲线在该点的**切线斜率**。

举例:$f(w) = w^2$ 在 $w = 3$ 处:
```
当 w 从 3 → 3.001 时,f(w) 从 9 → 9.006001,变化了 0.006001
近似变化率 = 0.006001 / 0.001 = 6.001 ≈ 6
```
精确:`df/dw = 2w = 2 * 3 = 6`

**直觉**:导数告诉你"参数 w 稍微往大了动一点,loss 会怎么变"。如果导数为正,w 增大会让 loss 增大 → 应该让 w **减小**。

#### 3. 偏导数(partial derivative)

多变量函数怎么办?比如 $L = (w \cdot x + b - y_{true})^2$,是关于 $w$、$b$、$x$、$y_{true}$ 四个变量的函数。我们只想知道 $L$ 对 $w$ 的导数(因为 $x$、$y_{true}$ 是数据,不能改)。

**偏导数** `∂L/∂w`:把 $w$ 以外的变量**当常数**,对 $w$ 求导。

举例:$L = (w \cdot x + b - y_{true})^2$
- 令 $u = w \cdot x + b - y_{true}$,则 $L = u^2$
- `∂L/∂u = 2u`
- `∂u/∂w = x`

#### 4. 梯度(gradient)

**定义**:**所有偏导数组成的向量**。

对 $L(w, b)$,梯度是:
```
∇L = [∂L/∂w, ∂L/∂b]
```

**关键性质**:**负梯度方向 = 函数下降最快的方向**。

直觉:站在山坡上,梯度告诉你"哪个方向最陡上",**反方向**就是"最陡下"。我们要最小化 loss → 沿**负梯度**更新参数:

```
w_new = w_old - lr * ∂L/∂w
```

`lr`(learning rate)就是"步长",控制每次更新多大。

#### 5. 链式法则(chain rule)

**复合函数求导法则**:一步一步乘起来。

举例:$y = u^2$,$u = w \cdot x$,求 $dy/dw$:
```
dy/dw = (dy/du) * (du/dw) = 2u * x = 2(w·x) * x
```

**这正是 autograd 的数学基础**。

神经网络就是个大复合函数 `loss = f(f(f(f(x))))`(几百万个 op 套起来)。手算链式法则 = 把所有 op 的局部导数乘起来。网络一大人手算不动 → 必须让 PyTorch **自动**做。

#### 6. 为什么不让 PyTorch 手算?——数值梯度 vs 反向传播

**数值梯度**(finite difference)思路:每个参数"稍微动一下"看 loss 变化:
```python
dw = (loss(w + eps) - loss(w - eps)) / (2 * eps)
```

- 优点:直观
- 缺点:**N 个参数要算 N 次**(每次还要 forward 两次)——几百万参数算一年

**反向传播**(backpropagation):
- 一次 forward 建计算图
- 一次 backward 把所有梯度**一次性**算出来
- 复杂度等价于一次 forward

**这就是为什么 PyTorch 要 autograd**:网络大的时候手算/数值算不可行,只有反向传播能搞定。

#### 7. 一个手算例子(带着手算走一遍,后面才能看懂 autograd)

最小线性回归:
```
数据:x = [1, 2, 3],y_true = [2, 4, 6]
参数:w(初始随机)、b(初始 0)
预测:y_pred = w*x + b
loss = mean((y_pred - y_true)^2)
```

手算 $\partial L/\partial w$:
```
令 err = y_pred - y_true
∂loss/∂err = 2*err / 3            ← mean 把 3 个样本平均了
∂err/∂y_pred = 1
∂y_pred/∂w = x

链式法则:∂loss/∂w = (∂loss/∂err) * (∂err/∂y_pred) * (∂y_pred/∂w)
                  = (2*err/3) * 1 * x
                  = (2/3) * (y_pred - y_true) * x
```

这一串人肉算就是 PyTorch 帮你做的事:
```python
loss = ((w*x + b - y_true)**2).mean()
loss.backward()         # 一行,PyTorch 内部把这一串全算完
print(w.grad, b.grad)   # ∂loss/∂w,∂loss/∂b 都有了
```

学完这一节再看 2.2.1,会发现 autograd 不是什么神奇黑科技——**就是链式法则 + 自动记录每一步 op**。

---

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

#### 正向模式 vs 反向模式(两种自动求导方向)

autograd 在内部支持两种**遍历计算图**的方向,功能一样,只是遍历策略不同:

| 模式 | 方向 | 适用 | DL 里用得多吗 |
|------|------|------|:------:|
| **反向模式**(reverse-mode) | 从 loss 反向往参数传 | loss=标量、参数 N 很大 | ✅ 主力 |
| **正向模式**(forward-mode) | 顺着前向,边算边求导 | 参数少、输出多的函数 | ❌ 罕见 |

PyTorch 默认就是**反向模式**——也就是 2.2.0.6 说的"反向传播"。原因上一节讲过:N 个参数、M=1 个 loss 时,反向模式只需一次扫,正向模式要 N 次扫,根本不可行。

> 极少数场景(物理仿真、某些元学习)需要"对参数求导"而不是"对 loss 求导",这时正向模式反而更快——这是 `torch.autograd.forward_ad` 做的事,本章用不上,知道有这条路就行。

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

> 这里 `loss_fn(y, target).mean()` 当 loss 用,常见选 `nn.MSELoss()`。MSE(均方误差)是回归任务的标准损失,公式 `mean((y_pred - y_true)²)` —— **预测值和真实值每个位置差的平方,再求平均**。直观:差距越大、平方后惩罚越重;多个样本平均掉 batch 维度,得到一个标量。02 章所有例子都用它,是因为它公式简单、求导容易、便于手算梯度。后续章节分类任务会换成 `nn.CrossEntropyLoss()`(1.4.5.4 提过),MSE 留作回归默认。

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

#### 这里累加的本质:是 PyTorch 的"行为",不是数学的"必须"

> ⚠️ **一个容易误解的点**:看到 `y1 = w**2`、`y2 = w**3` 共用同一个 `w`,有人会以为"w 出现在不同式子里所以梯度必须累加"。**这是错的。**

`w` 出现在多个式子 ≠ 梯度累加。**累加的真原因是:你调用了两次 `backward()`,而 PyTorch 在设计时选择把第二次的梯度加到第一次上面。**

对比下面两个例子,数学结构几乎一样,结果完全不同:

```python
# 例 A:两次 backward() → 累加
w = torch.tensor(2.0, requires_grad=True)
y1 = w ** 2; y1.backward()      # w.grad = 4
y2 = w ** 3; y2.backward()      # w.grad = 4 + 12 = 16  ← 累加
print(w.grad)                   # 16.0
```

```python
# 例 B:同一个 loss 表达式里 w 出现两次 → 不累加,而是按链式法则合并
w = torch.tensor(2.0, requires_grad=True)
y1 = w ** 2
y2 = w ** 3
loss = y1 + y2                  # loss = w^2 + w^3,w 在这里出现两次
loss.backward()                 # 一次 backward
print(w.grad)                   # 12.0 = 2w + 3w^2 = 4 + 12
```

注意例 B 里 `w` 同样在 `w**2` 和 `w**3` 两个式子里都出现,**但只调用了一次 `backward()`,结果不是累加,是一次链式法则求导出来的"全导数"**(全导数公式:`∂loss/∂w = ∂y1/∂w + ∂y2/∂w`,这条加法是数学上必须加的)。

那例 A 累加是怎么回事?它其实是"两个独立 backward 各自往 `w.grad` 上写一份"——只不过 PyTorch 的 `.grad` 默认用 `+=` 而不是 `=`。这种累加**纯粹是 PyTorch 的实现行为**,数学上没有任何"必须"。

#### 训练循环里为什么会累加?——这是设计选择,不是 bug

实际训练时,你**每读一个 batch 就会 backward 一次**,梯度按 batch 累加才有意义(等价于大 batch)。所以 PyTorch 让 `.grad` 默认累加。

但**如果你手动攒 batch**(gradient accumulation,显存不够时的常用技巧),你**就是希望它累加**——攒 N 个 batch 再 `optimizer.step()`,这时千万别 `zero_grad`。

#### 一句话总结

| 场景 | 累加吗 | 原因 |
|---|---|---|
| 同一个 loss, `w` 在里面出现多次 | 否 | 一次 `backward()` 按链式法则求全导数(数学合并) |
| 多次 `backward()` 之间没 `zero_grad` | 是 | PyTorch 的 `.grad` 默认用 `+=`(实现行为) |
| 训练循环忘 `zero_grad` | 是 | 每 batch 的梯度累到下一 batch,等效 batch 翻倍 |

**所以 03 那个例子的累加,是"两次 `backward()` 没清零"导致的,不是"`w` 出现在多个式子里"导致的**——这两件事容易混,但要分清。

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

**为什么要默认释放?** 反向传播算完后,中间节点的激活值、op 的反向函数都还要占显存。DL 模型一张图的中间激活能占好几 GB,算完就立刻释放是默认行为。

注意一个例外:**叶子节点的 `.grad` 默认会被保留**(因为这是优化器要拿的东西),中间节点的 grad_fn 和激活值才被释放。

所以 02 章的所有例子里,`w.grad` 反向之后还能 `print` 出来——叶子被保留;但如果你想对同一个 loss 二次 `.backward()`,会报错,因为中间图已经被清了。

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

进入 [`03-attention/doc/03-attention.md`](../../03-attention/doc/03-attention.md) 学维度变换与注意力矩阵 —— 把 01 章的 `reshape / transpose / matmul` 用到极致,补成完整的 scaled dot-product attention 和多头。
