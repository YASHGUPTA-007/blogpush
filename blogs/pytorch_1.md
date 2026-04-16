---
title: "PyTorch Tensors Deep Dive: dtypes, Device Movement, Memory Layout, and Broadcasting"
excerpt: "Master PyTorch tensors from the ground up — dtypes, CUDA device movement, memory layout, strides, and broadcasting rules that trip up every beginner."
author: "Soham Sharma"
category: "Technology"
tags: ["PyTorch", "Deep Learning", "Tensors", "Python", "Machine Learning"]
status: "published"
featuredImage: ""
---

Tensors are the backbone of PyTorch. Get them wrong and you'll waste hours debugging device mismatches, silent precision loss, and memory errors that only surface at training time. This post covers everything a working ML engineer needs to know about tensors — not the toy-tutorial version, but the real mechanics that matter in production code.

![PyTorch tensor operations visualized](https://pytorch.org/assets/images/pytorch-logo.png)

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

# This raises: RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long
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

**Multi-GPU scenarios**: When you have multiple GPUs, be explicit about which device you want:

```python
# Specific GPU selection
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
    # Creates mask on same device as input, no hardcoding
    batch_size = sequence_lengths.shape[0]
    positions = torch.arange(max_len, device=sequence_lengths.device)
    mask = positions.unsqueeze(0) < sequence_lengths.unsqueeze(1)
    return mask
```

## Memory Layout: Contiguous Tensors and Strides

This is where most engineers' understanding stops — and where bugs hide.

Every PyTorch tensor has a **storage** (a flat array in memory) and a **view** defined by its shape, strides, and storage offset. The stride of dimension `i` tells you how many elements to jump in the underlying storage to advance by 1 in that dimension.

```python
x = torch.arange(12).reshape(3, 4)
print(x)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

print(x.shape)    # torch.Size([3, 4])
print(x.stride()) # (4, 1)  — row-major (C-contiguous)
# To move one row: jump 4 elements. To move one column: jump 1 element.
```

**Contiguous tensors** have strides that match what you'd expect for their shape in row-major order. After operations like `transpose()`, `permute()`, or slicing, tensors may become non-contiguous:

```python
x = torch.arange(12).reshape(3, 4)
x_t = x.T   # transpose

print(x_t.shape)         # torch.Size([4, 3])
print(x_t.stride())      # (1, 4)  — NOT row-major anymore
print(x_t.is_contiguous())  # False

# Many operations require contiguous tensors
# Fix with .contiguous() — creates a new tensor with proper layout
x_t_c = x_t.contiguous()
print(x_t_c.stride())    # (3, 1)  — now row-major for shape [4, 3]
print(x_t_c.is_contiguous())  # True
```

When does non-contiguous layout cause problems? Operations like `view()` require contiguous memory. Use `reshape()` instead — it falls back to copying if needed:

```python
x = torch.arange(12).reshape(3, 4)
x_t = x.T

# This raises RuntimeError: view size is not compatible with input tensor's size and stride
# x_t.view(12)

# reshape() handles non-contiguous tensors silently (may or may not copy)
x_flat = x_t.reshape(12)  # works

# Explicit: check and fix
if not x_t.is_contiguous():
    x_t = x_t.contiguous()
x_flat = x_t.view(12)  # now safe
```

**Memory sharing**: operations like `view()`, `transpose()`, and slicing return tensors that share storage with the original. Modifying one modifies the other:

```python
x = torch.arange(6).reshape(2, 3)
y = x[0]   # slice — shares storage

y[0] = 99
print(x)
# tensor([[99,  1,  2],
#         [ 3,  4,  5]])

# To avoid aliasing, use .clone()
z = x[0].clone()
z[0] = 0   # doesn't affect x
```

## Broadcasting Rules

Broadcasting lets you perform operations on tensors with different shapes without explicitly expanding memory. It follows NumPy's rules, but the mechanics are worth understanding precisely to avoid shape bugs.

![Broadcasting rules illustrated for multi-dimensional arrays](https://numpy.org/doc/stable/_images/broadcasting_4.png)

The rules, applied right-to-left across dimensions:

1. If tensors have different numbers of dimensions, prepend 1s to the shape of the smaller one.
2. Dimensions of size 1 are "stretched" to match the other tensor's size in that dimension.
3. If sizes differ and neither is 1, raise an error.

```python
# Rule in action
a = torch.rand(3, 1, 5)   # shape [3, 1, 5]
b = torch.rand(   4, 5)   # shape [4, 5]

# Step 1: align dimensions from the right
# a: [3, 1, 5]
# b: [1, 4, 5]  ← prepend 1

# Step 2: broadcast
# dim 0: max(3, 1) = 3
# dim 1: max(1, 4) = 4
# dim 2: max(5, 5) = 5
# result: [3, 4, 5]

c = a + b
print(c.shape)  # torch.Size([3, 4, 5])
```

Common practical patterns:

```python
# Bias addition in a linear layer (manual implementation)
weight = torch.rand(256, 128)
bias = torch.rand(128)           # shape [128]
x = torch.rand(32, 256)          # batch of 32, input dim 256

output = x @ weight.T + bias    # [32, 128] + [128] → broadcasts to [32, 128]

# Normalization: subtract mean per sample
batch = torch.rand(32, 512)          # [batch, features]
means = batch.mean(dim=1, keepdim=True)  # [32, 1]
normalized = batch - means           # [32, 512] - [32, 1] broadcasts correctly

# Without keepdim, this would fail
means_wrong = batch.mean(dim=1)     # [32]
# batch - means_wrong would raise: the shapes [32, 512] and [32] aren't broadcast-compatible
```

**The keepdim trap**: always use `keepdim=True` when you plan to broadcast a reduction result back against the original tensor.

```python
# Computing softmax manually — demonstrates keepdim importance
logits = torch.rand(4, 10)   # [batch, classes]

max_logits = logits.max(dim=1, keepdim=True).values  # [4, 1]
shifted = logits - max_logits                          # [4, 10] - [4, 1] ✓

exp_shifted = shifted.exp()
sum_exp = exp_shifted.sum(dim=1, keepdim=True)        # [4, 1]
softmax = exp_shifted / sum_exp                        # [4, 10] ✓
```

### Advanced: Explicit Dimension Management with unsqueeze/squeeze

```python
a = torch.rand(5)       # shape [5]
b = torch.rand(3, 5)    # shape [3, 5]

# To add a to each row of b:
a_row = a.unsqueeze(0)  # [1, 5] — broadcast across rows
result = b + a_row       # [3, 5]

# Equivalent with None indexing
result = b + a[None, :]  # same thing

# To add a column vector:
a_col = a[:3].unsqueeze(1)  # [3, 1] — broadcast across columns
result = b + a_col           # [3, 5]
```

## Practical Checklist

Before shipping any tensor manipulation code, run through these checks:

```python
def validate_tensor(t, name="tensor"):
    print(f"{name}:")
    print(f"  dtype:        {t.dtype}")
    print(f"  device:       {t.device}")
    print(f"  shape:        {t.shape}")
    print(f"  strides:      {t.stride()}")
    print(f"  contiguous:   {t.is_contiguous()}")
    print(f"  memory (MB):  {t.element_size() * t.nelement() / 1e6:.2f}")
    print(f"  requires_grad:{t.requires_grad}")

model_weights = torch.rand(1024, 1024, dtype=torch.float32, device="cuda")
validate_tensor(model_weights, "model_weights")
```

The `element_size()` call returns bytes per element — critical for memory budgeting. A float32 tensor of shape `[1024, 1024]` takes 4 MB. The same tensor in float16 takes 2 MB. At model scale (billions of parameters), this arithmetic determines whether you can fit a model on a single GPU.

## Conclusion

PyTorch tensors are not just NumPy arrays on a GPU. Understanding dtypes prevents silent precision loss and type errors at embedding layers. Knowing how `.to()` works prevents device mismatch bugs. Grasping contiguity and strides saves you from mysterious `view()` failures. And broadcasting rules, once internalized, let you write expressive, allocation-free code.

Start every new model component by printing tensor shapes, dtypes, and devices at each step. The 30 seconds this costs upfront saves hours of debugging later.
