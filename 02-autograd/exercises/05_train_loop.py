"""练习 05:迷你训练循环(综合)

要求:
  1. 数据: y = 3x + 2 + 噪声,生成 100 个点
  2. 参数: w, b, requires_grad=True,初始为 0
  3. 训练 200 步,每步:
     - forward: y_pred = w*x + b
     - loss = MSE
     - backward + 手动 SGD 更新(lr=0.01)
     - zero_grad
  4. 每 20 步打印一次 w, b, loss
  5. 训练完打印最终学到的 w(应该接近 3)、b(应该接近 2)
"""
import torch

# --- 1. 准备数据 ---
# 固定随机种子,保证每次运行生成的噪声一样,方便对比实验结果
torch.manual_seed(0)

# 生成 100 个均匀分布在 [0, 10] 的 x,作为输入特征
x = torch.linspace(0, 10, 100)
y_true = 3 * x + 2 + torch.randn(100) * 0.5

# --- 2. 初始化模型参数 ---
# requires_grad=True 是关键 —— PyTorch 靠这个标志追踪梯度,没有它就 .backward() 不了
w = torch.tensor(5.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# 学习率:控制每一步参数更新的幅度,太大容易震荡,太小收敛慢
lr = 0.01

# 表头,方便对齐查看每一步参数和梯度的变化
print(f"{'step':>4} {'w':>8}  {'w.grad':>10} {'b':>8} {'b.grad':>10} {'loss':>10}")

# --- 3. 训练循环 ---
for step in range(1, 201):
    # --- 前向传播 (forward) ---
    # 用当前参数 w, b 对所有 x 做线性预测
    # y_pred 形状是 (100,),跟 y_true 对得上,后面才能逐元素算 MSE
    y_pred = w * x + b

    # 均方误差损失:MSE = mean((y_pred - y_true)^2)
    # PyTorch 会自动构建计算图,把这次 loss 跟 w、b 关联起来
    loss = ((y_pred - y_true) ** 2).mean()

    # --- 反向传播 (backward) ---
    # 从 loss 开始沿计算图反向求梯度,结果累加到 w.grad 和 b.grad
    # 注意:梯度是**累加**的,所以下一步更新前必须清零,否则会叠加历史梯度
    loss.backward()

    # 打印监控信息:每 20 步打一次当前参数值、梯度值、损失值
    # .item() 是把单元素 tensor 转成 Python 数字,方便 f-string 格式化
    # .grad 也是累加后的值,所以打印前不要清零 —— 我们要看的就是这一步算出来的梯度
    if step % 20 == 0:
        print(f"{step:>4} {w.item():>8.4f} {w.grad.item():>10.4f} "
              f"{b.item():>8.4f} {b.grad.item():>10.4f} {loss.item():>10.4f}")

    # --- 参数更新 (手动 SGD) ---
    # with torch.no_grad(): 表示这一步不参与梯度计算
    # 因为我们只是想"应用"梯度更新参数,而不是把更新动作本身也记进计算图
    # 写法 w -= lr * w.grad 等价于 w = w - lr * w.grad,但原地操作更省内存
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # --- 梯度清零 ---
    # PyTorch 的 .grad 是累加器,下次 backward 之前必须清零,
    # 否则这次的梯度会跟下次累加,导致参数更新方向错误
    # 用 zero_() 是原地操作;也可以用 w.grad = None(更省内存,但功能略有不同)
    w.grad.zero_()
    b.grad.zero_()

# --- 4. 检查最终结果 ---
# 真实规律是 y = 3x + 2,如果模型学得对,w 应该接近 3,b 应该接近 2
print(f"最终学到的 w: {w.item():>.4f}(应接近3), b: {b.item():>.4f}(应接近2)")
