---
title: "PyTorch Tensors Deep Dive: dtypes, Device Movement, Memory Layout, and Broadcasting"
excerpt: "Master PyTorch tensors from the ground up — dtypes, CUDA device movement, memory layout, strides, and broadcasting rules that trip up every beginner."
author: "Soham Sharma"
category: "Technology"
tags: ["PyTorch", "Deep Learning", "Tensors", "Python", "Machine Learning"]
status: "published"
featuredImage: "https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1200&auto=format&fit=crop&q=80"
---

Tensors are the backbone of PyTorch. Get them wrong and you'll waste hours debugging device mismatches, silent precision loss, and memory errors that only surface at training time. This post covers everything a working ML engineer needs to know about tensors — not the toy-tutorial version, but the real mechanics that matter in production code.

![Code running on multiple screens in a dark development environment](https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200&auto=format&fit=crop&q=80)

## Data Types: Choosing the Right dtype

Every tensor has a dtype. Pick the wrong one and you'll either lose precision silently or double your memory footprint for no reason.

```python
import torch

# The default floating-point dtype is float32
x = torch.tensor([1.0, 2.0, 3.0])
print(x.dtype)  # torch.float32

# Explicit dtype specification
x_fp64 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
x_fp16 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
x_bf16 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)

# Integer types
indices = torch.tensor([0, 1, 2], dtype=torch.int64)   # standard for indices
mask = torch.tensor([True, False, True], dtype=torch.bool)
```

The dtype hierarchy matters when mixing types in operations:

| dtype | Bits | Range | Use case |
|---|---|---|---|
| `torch.float32` | 32 | ~±3.4×10³⁸ | Default training |
| `torch.float64` | 64 | ~±1.8×10³⁰⁸ | Scientific computation |
| `torch.float16` | 16 | ~±65504 | Mixed-precision inference |
| `torch.bfloat16` | 16 | ~±3.4×10³⁸ | TPUs, Ampere+ GPUs |
| `torch.int64` | 64 | ±9.2×10¹⁸ | Indices, token IDs |
| `torch.int32` | 32 | ±2.1×10⁹ | Counts, positions |
| `torch.bool` | 8 | True/False | Masks, attention masks |

**The bfloat16 vs float16 decision** is critical: float16 has higher precision (10-bit mantissa vs 7-bit) but a much narrower range (max ~65504). bfloat16 matches float32's exponent range, making it less prone to overflow during training. On A100/H100 GPUs and all TPUs, prefer bfloat16 for training. Use float16 for inference on older hardware.

```python
# Type conversion
x = torch.rand(3, 3)          # float32
x_half = x.half()             # float16, alias for .to(torch.float16)
x_bf16 = x.bfloat16()         # bfloat16
x_int = x.int()               # int32, truncates decimal

# Safe conversion with explicit dtype
x_converted = x.to(dtype=torch.float64)

# Check dtype before operations that require specific types
assert x.dtype == torch.float32, f"Expected float32, got {x.dtype}"
```

A common trap: embedding layers require `int64` (Long) indices. Passing `int32` raises a RuntimeError that looks like a shape error at first glance.

```python
embedding = torch.nn.Embedding(1000, 64)
indices_wrong = torch.tensor([1, 2, 3], dtype=torch.int32)
indices_right = torch.tensor([1, 2, 3], dtype=torch.int64)

# This raises: RuntimeError: Expected tensor for argument #1 'indices'
# to have scalar type Long
# embedding(indices_wrong)

# This works
output = embedding(indices_right)  # shape: [3, 64]
```

## Device Movement: CPU ↔ CUDA

Device mismatch is the most common runtime error for PyTorch beginners. The rule is simple: **all operands in a computation must be on the same device**.

```python
import torch

# Create tensors on different devices
cpu_tensor = torch.rand(3, 3)
print(cpu_tensor.device)  # cpu

# Move to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_tensor = cpu_tensor.to(device)
print(gpu_tensor.device)  # cuda:0

# Alternative syntax
gpu_tensor = cpu_tensor.cuda()   # assumes cuda:0
cpu_back = gpu_tensor.cpu()      # back to CPU

# Move with dtype change in one call
gpu_fp16 = cpu_tensor.to(device=device, dtype=torch.float16)
```

The `.to()` method is idempotent — calling `.to("cuda")` on a tensor already on CUDA returns the same tensor without a copy. This matters for performance: don't worry about guarding against redundant `.to()` calls.

```python
# Best practice: define device once at top of script
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create tensors directly on device (faster than CPU-then-move)
x = torch.rand(1000, 1000, device=device)
y = torch.rand(1000, 1000, device=device)
z = x @ y  # matrix multiply on GPU

# Moving model and data to same device
model = MyModel().to(device)
batch = batch.to(device)  # move input batch in training loop
```

**Multi-GPU scenarios**: when you have multiple GPUs, be explicit about which device you want:

```python
device_0 = torch.device("cuda:0")
device_1 = torch.device("cuda:1")

x = torch.rand(100, device=device_0)
y = torch.rand(100, device=device_1)

# This raises RuntimeError: Expected all tensors to be on the same device
# z = x + y

# Explicit move before operation
z = x + y.to(device_0)
```

A practical pattern: use `tensor.device` to create new tensors on the same device as an existing one, avoiding hardcoded device strings:

```python
def create_mask(sequence_lengths, max_len):
    # Creates mask on same device as input — no hardcoded device strings
    positions = torch.arange(max_len, device=sequence_lengths.device)
    mask = positions.unsqueeze(0) < sequence_lengths.unsqueeze(1)
    return mask
```

## Memory Layout: Contiguous Tensors and Strides

Every PyTorch tensor has a **storage** (a flat array in memory) and a **view** defined by its shape, strides, and storage offset. The stride of dimension `i` tells you how many elements to skip in storage to advance by 1 in that dimension.

```python
x = torch.arange(12).reshape(3, 4)
print(x.stride())       # (4, 1)  — row-major (C-contiguous)
print(x.is_contiguous()) # True

x_t = x.T              # transpose
print(x_t.stride())     # (1, 4)  — NOT row-major
print(x_t.is_contiguous())  # False

# view() requires contiguous memory — use reshape() instead
x_flat = x_t.reshape(12)   # works, may copy
# x_t.view(12)             # raises RuntimeError

# Or fix explicitly
x_t_c = x_t.contiguous()
x_flat = x_t_c.view(12)    # now safe
```

**Memory sharing**: slices and `view()` share storage with the original. Modifying one modifies the other:

```python
x = torch.arange(6).reshape(2, 3)
y = x[0]   # shares storage

y[0] = 99
print(x)   # tensor([[99,  1,  2], [ 3,  4,  5]])

# To avoid aliasing
z = x[0].clone()
z[0] = 0   # doesn't affect x
```

## Broadcasting Rules

Broadcasting lets you perform operations on tensors with different shapes without explicitly expanding memory. Applied right-to-left:

1. Prepend 1s to the shape of the smaller tensor.
2. Dimensions of size 1 stretch to match the other tensor.
3. If sizes differ and neither is 1, error.

```python
a = torch.rand(3, 1, 5)   # [3, 1, 5]
b = torch.rand(   4, 5)   # [4, 5] → [1, 4, 5] after prepend

c = a + b
print(c.shape)  # [3, 4, 5]
```

The most common broadcasting mistake is forgetting `keepdim=True` when reducing and then operating on the result:

```python
batch = torch.rand(32, 512)

# Wrong — shapes [32, 512] and [32] don't broadcast
means = batch.mean(dim=1)         # [32]
# normalized = batch - means      # RuntimeError

# Correct — keepdim preserves the dimension for broadcasting
means = batch.mean(dim=1, keepdim=True)  # [32, 1]
normalized = batch - means               # [32, 512] ✓
```

![Abstract visualization of multi-dimensional data arrays and tensor mathematics](https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=1200&auto=format&fit=crop&q=80)

### Explicit Dimension Control with unsqueeze

```python
a = torch.rand(5)       # [5]
b = torch.rand(3, 5)    # [3, 5]

# Add a to each row of b
result = b + a.unsqueeze(0)   # [3,5] + [1,5] → [3,5]

# Equivalent using None indexing
result = b + a[None, :]       # same

# Add a column vector
col = a[:3].unsqueeze(1)      # [3,1]
result = b + col              # [3,5] ✓
```

## Practical Checklist

Before shipping any tensor manipulation code:

```python
def validate_tensor(t, name="tensor"):
    print(f"{name}: dtype={t.dtype}, device={t.device}, "
          f"shape={t.shape}, contiguous={t.is_contiguous()}, "
          f"memory={t.element_size() * t.nelement() / 1e6:.2f}MB")

model_weights = torch.rand(1024, 1024, dtype=torch.float32)
validate_tensor(model_weights, "model_weights")
# model_weights: dtype=torch.float32, device=cpu, shape=torch.Size([1024, 1024]),
# contiguous=True, memory=4.19MB
```

`element_size()` returns bytes per element — float32 is 4 bytes, float16 is 2. At model scale, the difference between float32 and float16 determines whether you can fit a model on a single GPU.

## Conclusion

Understanding PyTorch dtypes prevents silent precision loss and index type errors. Knowing how `.to()` works prevents device mismatch crashes. Grasping contiguity and strides saves you from mysterious `view()` failures. And broadcasting, once internalized, lets you write expressive, allocation-free code. Print tensor shapes, dtypes, and devices at every step when building a new component — 30 seconds upfront saves hours of debugging.
