"""02-4 torch.no_grad() 推理模式

对应 autograd 章节第 4 条(重点),也对应"显存防爆"目标 10。
"""
import torch

w = torch.randn(3, 3, requires_grad=True)

# 正常模式:会建计算图
y = (w * 2).sum()
print(f"正常 forward: y.requires_grad = {y.requires_grad}")
print(f"  y.grad_fn = {y.grad_fn}")

# 推理模式:用 no_grad 包起来
with torch.no_grad():
    y_inf = (w * 2).sum()
    print(f"\nno_grad 内:")
    print(f"  y_inf.requires_grad = {y_inf.requires_grad}  # False")
    print(f"  y_inf.grad_fn = {y_inf.grad_fn}  # None")

# 替代写法:tensor.detach()
y_det = (w * 2).sum().detach()
print(f"\ndetach() 后:")
print(f"  y_det.requires_grad = {y_det.requires_grad}")

# 显存上的差别
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    big = torch.randn(1000, 1000, device="cuda", requires_grad=True)
    torch.cuda.synchronize()
    mem_with_grad = torch.cuda.memory_allocated() / 1024**2
    print(f"\n带梯度的 1000x1000 占显存: {mem_with_grad:.2f} MB")

    with torch.no_grad():
        big2 = torch.randn(1000, 1000, device="cuda")  # 不会建图
    print(f"no_grad 内同样大小: {torch.cuda.memory_allocated() / 1024**2:.2f} MB (增量)")
