---
title: >-
  PyTorch Tensors Deep Dive: dtypes, Device Movement, Memory Layout, and
  Broadcasting
excerpt: >-
  Tensors are the foundation of every PyTorch model. Master dtypes, device
  movement, memory layout, and broadcasting rules to eliminate hours of
  debugging.
author: Soham Sharma
authorName: Soham Sharma
category: AI
tags:
  - PyTorch
  - Tensors
  - Deep Learning
  - Python
  - GPU
status: published
featuredImage: >-
  https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200&auto=format&fit=crop&q=80
colab_notebook: >-
  https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/pytorch/pytorch_1.ipynb
series_id: pytorch-mastery
series_slug: pytorch-mastery
series_title: 'PyTorch Mastery: From Tensors to Production'
difficulty: beginner
week: null
day: 1
tools:
  - PyTorch
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/pytorch/pytorch_1.ipynb)


Tensors are the backbone of PyTorch. Get them wrong and you'll waste hours debugging device mismatches, silent dtype promotions, and memory layout errors that only surface at batch boundaries. This post strips away the abstraction and shows you exactly how tensors work under the hood — not because it's academically interesting, but because this knowledge directly translates to fewer bugs and faster models.

## What a Tensor Actually Is

A PyTorch tensor is a multi-dimensional array backed by a contiguous (or strided) block of memory. Unlike a NumPy array, a tensor tracks:

- Its **dtype** (the numeric type of each element)
- Its **device** (CPU or a specific CUDA device)
- Its **memory layout** (contiguous vs. strided)
- Its **gradient requirement** (whether autograd should track operations on it)

The combination of these four properties determines what operations are valid and how fast they run.

![PyTorch tensor memory layout diagram](https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=1200&auto=format&fit=crop&q=80)

## dtypes: Choosing the Right Numeric Type

PyTorch provides a range of dtypes. Choosing the wrong one is one of the most common sources of silent bugs — a model that trains but converges poorly because weights silently became integers.

The most important dtypes in practice:

| dtype | PyTorch constant | Bits | Use case |
|---|---|---|---|
| 32-bit float | `torch.float32` | 32 | Default training dtype |
| 16-bit float | `torch.float16` | 16 | Mixed-precision training (Volta+) |
| bfloat16 | `torch.bfloat16` | 16 | Mixed-precision on Ampere, TPUs |
| 64-bit float | `torch.float64` | 64 | Scientific computing, rarely ML |
| 32-bit int | `torch.int32` | 32 | Index tensors, counts |
| 64-bit int | `torch.int64` | 64 | Default integer dtype |
| boolean | `torch.bool` | 8 | Masks |

The following code demonstrates how to create tensors with specific dtypes and how PyTorch reports their properties.

```python
import torch

# Default dtype is float32
x = torch.tensor([1.0, 2.0, 3.0])
print(f"dtype: {x.dtype}")
print(f"shape: {x.shape}")
print(f"device: {x.device}")
```

**Output:**
```text
dtype: torch.float32
shape: torch.Size([3])
device: cpu
```

`torch.float32` is the default because it offers the right balance of precision and memory for most ML workloads. `torch.float64` doubles the memory footprint with little practical benefit for neural networks.

### Explicit dtype control

Relying on Python literal types to infer dtype leads to surprises. Always specify dtype explicitly when the type matters.

```python
import torch

# Python int literal → torch.int64 by default
a = torch.tensor([1, 2, 3])

# Python float literal → torch.float32 by default
b = torch.tensor([1.0, 2.0, 3.0])

# Explicit dtype
c = torch.tensor([1, 2, 3], dtype=torch.float32)
d = torch.tensor([1, 2, 3], dtype=torch.int8)

print(a.dtype, b.dtype, c.dtype, d.dtype)
```

**Output:**
```text
torch.int64 torch.float32 torch.float32 torch.int8
```

Note that `a` is `int64` while `c` is `float32` despite having identical values. Passing `a` into a linear layer will throw a `RuntimeError: expected scalar type Float` — a common beginner mistake.

### Type casting

Use `.to(dtype)` or the shorthand type methods to cast:

```python
import torch

x = torch.tensor([1, 2, 3], dtype=torch.int64)

# Preferred: .to()
x_float = x.to(torch.float32)

# Shorthand (equivalent)
x_float2 = x.float()  # → float32
x_half = x.half()     # → float16

print(x_float.dtype, x_float2.dtype, x_half.dtype)
```

**Output:**
```text
torch.float32 torch.float32 torch.float16
```

### Gotcha: dtype promotion in arithmetic

When you mix dtypes in an operation, PyTorch promotes to the "wider" type. This is usually correct but can silently inflate memory usage.

```python
import torch

a = torch.tensor([1.0], dtype=torch.float16)
b = torch.tensor([2.0], dtype=torch.float32)

result = a + b
print(result.dtype)  # promoted to float32
```

**Output:**
```text
torch.float32
```

> **Pitfall:** If your loss computation mixes `float16` activations with `float32` parameters (common in naive mixed-precision setups), you can end up with unintended promotions that negate memory savings. Use `torch.cuda.amp` for properly scoped mixed precision instead of manual casting.

## Device Movement: CPU ↔ GPU

PyTorch tensors live on a specific device. Operations between tensors on different devices fail immediately with a `RuntimeError`. Device management is explicit — nothing moves automatically.

```python
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

x = torch.tensor([1.0, 2.0, 3.0])
print(f"x device: {x.device}")

# Move to GPU
x_gpu = x.to(device)
print(f"x_gpu device: {x_gpu.device}")
```

**Output:**
```text
Using device: cuda
x device: cpu
x_gpu device: cuda:0
```

> Note: Output shows `cpu` if no GPU is available on your machine.

The `cuda:0` suffix identifies the specific GPU index. On multi-GPU machines you may have `cuda:0`, `cuda:1`, etc.

### Creating tensors directly on device

Avoid the pattern of creating on CPU then moving to GPU — it wastes a memory allocation and a copy.

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Slow: CPU allocation + copy
x_slow = torch.zeros(1000, 1000).to(device)

# Fast: direct allocation on target device
x_fast = torch.zeros(1000, 1000, device=device)

print(x_fast.device)
```

**Output:**
```text
cuda:0
```

> Note: Output is `cpu` if no GPU is present.

For all factory functions (`torch.zeros`, `torch.ones`, `torch.randn`, `torch.empty`), pass `device=` to create directly on the target device.

### Moving back to CPU for NumPy interop

NumPy cannot access GPU memory. To convert a GPU tensor to NumPy, you must first move it to CPU and detach it from the computation graph.

```python
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)

# This would fail on GPU:
# arr = x.numpy()  # RuntimeError

# Correct pattern:
arr = x.detach().cpu().numpy()
print(arr, type(arr))
```

**Output:**
```text
[1. 2. 3.] <class 'numpy.ndarray'>
```

The chain `.detach().cpu().numpy()` is idiomatic. `detach()` removes gradient tracking, `cpu()` moves the data, `numpy()` wraps the underlying buffer.

> **Pitfall:** Calling `.numpy()` on a tensor that still has `requires_grad=True` raises `RuntimeError: Can't call numpy() on Tensor that requires grad`. Always detach before converting.

## Memory Layout: Contiguous vs. Strided

This is the least understood aspect of tensors and the source of some of the most confusing errors.

Every tensor has two things: **storage** (a flat, contiguous block of memory) and **metadata** (shape, strides, storage offset) that describe how to index into that storage. Strides tell PyTorch how many elements to skip in storage to advance by one step along each dimension.

```python
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]], dtype=torch.float32)

print(f"shape:   {x.shape}")
print(f"strides: {x.stride()}")
print(f"is_contiguous: {x.is_contiguous()}")
```

**Output:**
```text
shape:   torch.Size([2, 3])
strides: (3, 1)
is_contiguous: True
```

The strides `(3, 1)` mean: moving along dimension 0 (rows) skips 3 elements in storage; moving along dimension 1 (columns) skips 1 element. This is standard row-major (C-style) layout.

### How transpose breaks contiguity

```python
import torch

x = torch.randn(3, 4)
print(f"x strides: {x.stride()}, contiguous: {x.is_contiguous()}")

xt = x.T  # or x.transpose(0, 1)
print(f"xt strides: {xt.stride()}, contiguous: {xt.is_contiguous()}")
```

**Output:**
```text
x strides: (4, 1), contiguous: True
xt strides: (1, 4), contiguous: False
```

`x.T` does not copy memory — it just swaps the strides in the metadata. The data layout in memory is unchanged, but the strides now describe column-major (Fortran-style) access. Accessing elements row-by-row now skips 4 elements per step — cache-unfriendly.

### When non-contiguous tensors cause errors

Some operations require contiguous memory:

```python
import torch

x = torch.randn(3, 4).T  # non-contiguous

try:
    x.view(12)  # view() requires contiguous
except RuntimeError as e:
    print(f"Error: {e}")

# Fix: call .contiguous() first
x_c = x.contiguous()
print(x_c.view(12).shape)
```

**Output (raises):**
```text
Error: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

**Output:**
```text
torch.Size([12])
```

`.contiguous()` forces a memory copy into a new contiguous block. Use `.reshape()` instead of `.view()` when you're not sure about contiguity — `.reshape()` returns a view if possible and copies only when necessary.

![GPU memory layout visualization](https://images.unsplash.com/photo-1639762681485-074b7f938ba0?w=1200&auto=format&fit=crop&q=80)

## Broadcasting Rules

Broadcasting lets PyTorch perform element-wise operations on tensors with different shapes, following well-defined rules. Getting broadcasting wrong is silent — the result will be wrong-shaped or wrong-valued, but no error is thrown.

The rules, applied right-to-left across dimensions:

1. If tensors have different numbers of dimensions, prepend 1s to the smaller shape.
2. Dimensions of size 1 are stretched to match the other tensor's size.
3. If sizes differ and neither is 1, raise an error.

```python
import torch

a = torch.tensor([[1.0], [2.0], [3.0]])  # shape (3, 1)
b = torch.tensor([10.0, 20.0, 30.0])    # shape (3,) → (1, 3) after prepend

result = a + b  # broadcasts to (3, 3)
print(f"a shape: {a.shape}")
print(f"b shape: {b.shape}")
print(f"result shape: {result.shape}")
print(result)
```

**Output:**
```text
a shape: torch.Size([3, 1])
b shape: torch.Size([3])
result shape: torch.Size([3, 3])
tensor([[11., 21., 31.],
        [12., 22., 32.],
        [13., 23., 33.]])
```

`b` is first treated as shape `(1, 3)` (prepend rule), then both tensors are stretched: `a` from `(3,1)` to `(3,3)`, `b` from `(1,3)` to `(3,3)`. The result is an outer-sum.

### A practical broadcasting example: batch normalization by hand

```python
import torch

# batch of 4 samples, 3 features each
batch = torch.randn(4, 3)

mean = batch.mean(dim=0)   # shape (3,)
std  = batch.std(dim=0)    # shape (3,)

# mean/std broadcast over batch dimension automatically
normalized = (batch - mean) / (std + 1e-8)

print(f"batch:      {batch.shape}")
print(f"mean:       {mean.shape}")
print(f"normalized: {normalized.shape}")
print(f"col means after norm: {normalized.mean(dim=0).round(decimals=4)}")
```

**Output:**
```text
batch:      torch.Size([4, 3])
mean:       torch.Size([3])
normalized: torch.Size([4, 3])
col means after norm: tensor([0., 0., 0.])
```

> Note: Exact values vary by random seed.

`mean` has shape `(3,)` which broadcasts against `batch` shape `(4, 3)` — the subtraction is applied to each of the 4 rows. This is the core of many normalization operations.

### Gotcha: unintended broadcasting from wrong shapes

```python
import torch

# Intended: add a bias to each of 4 samples
bias = torch.tensor([1.0, 2.0, 3.0])   # shape (3,) — correct
wrong_bias = torch.tensor([[1.0], [2.0], [3.0]])  # shape (3,1) — wrong!

x = torch.zeros(4, 3)

correct = x + bias        # (4,3) + (3,) → (4,3) ✓
wrong   = x + wrong_bias  # (4,3) + (3,1) → (4,3) but adds wrong values!

print(f"correct shape: {correct.shape}")
print(f"wrong shape:   {wrong.shape}")
print(f"correct[0]:    {correct[0]}")
print(f"wrong[0]:      {wrong[0]}")
```

**Output:**
```text
correct shape: torch.Size([4, 3])
wrong shape:   torch.Size([4, 3])
correct[0]:    tensor([1., 2., 3.])
wrong[0]:      tensor([1., 1., 1.])
```

Both produce a `(4,3)` tensor — no error — but `wrong` adds column-wise instead of element-wise. Always `print(tensor.shape)` when debugging unexpected values in batched operations.

## Practical Checklist

Before moving on to building models, these tensor hygiene habits will save you significant debugging time:

- **Always specify `device=` on factory functions** rather than creating on CPU and moving.
- **Check `.dtype` when a loss returns NaN** — a common cause is integer tensors passed to float operations.
- **Use `.reshape()` over `.view()`** unless you specifically need the contiguity guarantee.
- **`print(tensor.shape)` liberally** — shape mismatches account for the majority of runtime errors in new PyTorch code.
- **Use `torch.cuda.is_available()` checks** rather than hardcoding `"cuda"` — your code should run on CPU too.

## Conclusion

Tensors are not magic arrays — they are strided views into typed, device-specific memory. Understanding dtypes prevents silent precision bugs. Understanding device management prevents cross-device errors. Understanding memory layout explains why `view()` sometimes fails and why transposes can be slow. And understanding broadcasting lets you write vectorized code without introducing shape bugs that hide behind correct-looking output shapes.

The next post in this series covers autograd internals — the computation graph that sits behind every tensor operation you've just learned.
