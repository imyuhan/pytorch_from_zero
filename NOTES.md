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
