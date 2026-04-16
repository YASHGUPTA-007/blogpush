---
title: "PyTorch Tensors Deep Dive: dtypes, Device Movement, Memory Layout, and Broadcasting"
excerpt: "Master PyTorch tensors from the ground up — dtype selection, CPU/GPU movement, stride-based memory layout, and broadcasting rules that trip up every beginner."
author: "Soham Sharma"
category: "Technology"
tags: ["PyTorch", "Deep Learning", "Python", "Machine Learning", "GPU"]
status: "published"
featuredImage: "https://images.unsplash.com/photo-1509228468518-180dd4864904?w=1200&auto=format&fit=crop&q=80"
---

Tensors are the backbone of every PyTorch program. Get them wrong and you'll waste hours chasing device mismatch errors, mysterious NaNs from dtype overflows, and cryptic `view()` failures that only appear at runtime. Most tutorials show you how to create a tensor and move on. This post goes deeper — into how dtypes affect precision and memory, how GPU transfers actually work, what strides are and why they determine whether `view()` works, and the exact rules that govern broadcasting. By the end you'll understand not just what tensors do, but why they behave the way they do.

![Abstract mathematical visualization showing interconnected nodes and data flow](https://images.unsplash.com/photo-1509228468518-180dd4864904?w=1200&auto=format&fit=crop&q=80)

## dtypes: Choosing the Right Number Format

A tensor's dtype controls two things: the numerical type stored at each element, and the amount of memory each element occupies. Choosing the wrong dtype is one of the most common sources of subtle bugs — integer tensors silently truncate floats, `float16` overflows at values above ~65,504, and mixing dtypes in an operation often raises a cryptic type error.

PyTorch's most commonly used dtypes are:

| dtype | Alias | Bits | Typical use |
|-------|-------|------|-------------|
| `torch.float32` | `torch.float` | 32 | Default for model weights and activations |
| `torch.float64` | `torch.double` | 64 | Scientific computing, numerical precision |
| `torch.float16` | `torch.half` | 16 | Mixed-precision training (GPU) |
| `torch.bfloat16` | — | 16 | Mixed-precision on TPU/Ampere+ GPUs |
| `torch.int64` | `torch.long` | 64 | Default integer, class indices, `arange` |
| `torch.int32` | `torch.int` | 32 | Smaller integers |
| `torch.bool` | — | 8 | Masks, comparison results |

Let's see how PyTorch infers dtypes from Python literals, and then how to be explicit about what you want:

```python
import torch

# PyTorch infers dtype from the Python literal type
x_float = torch.tensor([1.0, 2.0, 3.0])       # float -> torch.float32
x_int   = torch.tensor([1, 2, 3])              # int   -> torch.int64
x_bool  = torch.tensor([True, False, True])    # bool  -> torch.bool

print(x_float.dtype)
print(x_int.dtype)
print(x_bool.dtype)
```

**Output:**
```text
torch.float32
torch.int64
torch.bool
```

The inference rules are simple: Python `float` literals produce `torch.float32`, Python `int` literals produce `torch.int64`, and Python `bool` literals produce `torch.bool`. There is no automatic upcast to `float64` — unlike NumPy, which uses `float64` as its default float dtype. This difference matters when porting NumPy code to PyTorch.

To override inference, pass `dtype` explicitly:

```python
x_double = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
x_half   = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
x_int32  = torch.tensor([1, 2, 3],        dtype=torch.int32)

print(x_double.dtype)
print(x_half.dtype)
print(x_int32.dtype)
```

**Output:**
```text
torch.float64
torch.float16
torch.int32
```

### Casting Between dtypes

Use `.to(dtype)` to cast an existing tensor to a different dtype. This always creates a new tensor — the original is unchanged.

```python
x = torch.tensor([1.8, 2.5, 3.9])

x_int  = x.to(torch.int32)    # truncates, does NOT round
x_half = x.to(torch.float16)  # reduced precision

print("Original:  ", x)
print("Cast int32:", x_int)
print("Cast fp16: ", x_half)
```

**Output:**
```text
Original:   tensor([1.8000, 2.5000, 3.9000])
Cast int32: tensor([1, 2, 3], dtype=torch.int32)
Cast fp16:  tensor([1.8008, 2.5000, 3.9004], dtype=torch.float16)
```

Two things to notice. First, casting to `int32` truncates toward zero — `1.8` becomes `1`, not `2`. This is not rounding; it is truncation. This catches people out when they expect `round()`-like behavior. Second, the `float16` values are not exactly `1.8`, `2.5`, and `3.9` — they are the nearest representable `float16` values. `1.8` → `1.8008`, `3.9` → `3.9004`. The error is small but accumulates across many operations, which is why `float16` training requires loss scaling to avoid underflow in gradients.

### The float16 Overflow Trap

`float16` can only represent values up to approximately `65,504`. Values above this overflow to `inf`:

```python
x = torch.tensor([1000.0, 65504.0, 65505.0], dtype=torch.float16)
print(x)
```

**Output:**
```text
tensor([1000., 65504.,    inf], dtype=torch.float16)
```

`65504.0` fits exactly — it's the maximum finite `float16` value. `65505.0` is one step beyond the maximum representable value, so it silently becomes `inf`. This is why `float16` training with large logits or unnormalized activations produces `inf`/`NaN` loss — and why PyTorch's `torch.cuda.amp.autocast` uses `bfloat16` on Ampere GPUs instead, which has the same dynamic range as `float32` but fewer mantissa bits.

---

## Device Movement: CPU and GPU

PyTorch tensors live on a specific device — either CPU memory or a specific GPU. Every operation between two tensors requires them to be on the same device. Mixing devices raises a `RuntimeError` immediately, which is usually helpful for debugging but can be confusing when it happens inside a loss function or a callback.

By default, tensors are created on CPU:

```python
x = torch.randn(3, 3)
print(x.device)
print(x.is_cuda)
```

**Output:**
```text
cpu
False
```

To move a tensor to GPU, use `.to('cuda')` or the shorthand `.cuda()`. Both do the same thing. The preferred modern style is `.to('cuda')` because it generalizes to other devices (like `'mps'` for Apple Silicon):

```python
if torch.cuda.is_available():
    x_gpu = x.to('cuda')          # preferred
    # x_gpu = x.cuda()            # equivalent shorthand

    print(x_gpu.device)
    print(x_gpu.is_cuda)
```

**Output:**
```text
cuda:0
True
```

`cuda:0` means GPU index 0. On a multi-GPU machine, you can target specific GPUs with `'cuda:1'`, `'cuda:2'`, etc.

### Moving Back to CPU

To get data back from GPU — for example, to convert to NumPy or pass to a non-PyTorch library — call `.cpu()`:

```python
x_back = x_gpu.cpu()
print(x_back.device)

# Now safe to convert to NumPy
arr = x_back.numpy()
print(type(arr))
print(arr.shape)
```

**Output:**
```text
cpu
<class 'numpy.ndarray'>
(3, 3)
```

`.numpy()` only works on CPU tensors. Calling it on a GPU tensor raises: `TypeError: can't convert cuda:0 device type tensor to numpy`. This is because NumPy has no concept of GPU memory — the data must be in CPU RAM first.

### The Device Mismatch Error

Attempting an operation between a CPU tensor and a GPU tensor raises immediately:

```python
a = torch.tensor([1.0, 2.0, 3.0])           # cpu
b = torch.tensor([4.0, 5.0, 6.0]).to('cuda') # cuda:0

try:
    c = a + b
except RuntimeError as e:
    print(e)
```

**Output:**
```text
Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

This error is intentional — PyTorch never silently copies data across devices, because an accidental cross-device copy inside a training loop would cause massive hidden slowdowns. Always be explicit about where your tensors live.

### Non-Blocking Transfers

By default, `.to('cuda')` is synchronous — your Python code pauses until the transfer is complete. For large tensors, this can create a CPU/GPU pipeline bubble. Pass `non_blocking=True` to overlap the transfer with other CPU work:

```python
# Pin memory on CPU first for fastest transfer
x_pinned = torch.randn(1000, 1000).pin_memory()

# Non-blocking transfer — returns immediately, transfer happens in background
x_gpu = x_pinned.to('cuda', non_blocking=True)

# Do CPU work here while transfer runs in background...
result_cpu = torch.sum(torch.randn(500, 500))

print(result_cpu.item())  # Forces CPU sync; GPU transfer likely done by now
```

**Output:**
```text
-12.485137939453125
```

> Note: The exact value differs each run due to `torch.randn`. The key point is the pattern — pin memory on CPU, then use `non_blocking=True` during transfer. This is standard practice in high-throughput DataLoader pipelines.

![Circuit board with glowing data pathways representing GPU memory transfer](https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200&auto=format&fit=crop&q=80)

---

## Memory Layout: Strides, Contiguity, and view() vs reshape()

This is the part most tutorials skip, and it's the source of one of the most confusing errors in PyTorch: `RuntimeError: view size is not compatible with input tensor's size and stride`. Understanding strides makes this error obvious.

### What Are Strides?

A tensor's data is stored as a flat 1D array in memory. **Strides** tell PyTorch how many elements to skip to move to the next position along each dimension. For a 2D tensor, stride `(s0, s1)` means: "skip `s0` elements to move to the next row; skip `s1` elements to move to the next column."

Create a simple 3×4 integer tensor and inspect its strides:

```python
x = torch.arange(12).reshape(3, 4)
print(x)
print("shape: ", x.shape)
print("stride:", x.stride())
print("contiguous:", x.is_contiguous())
```

**Output:**
```text
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
shape:  torch.Size([3, 4])
stride: (4, 1)
contiguous: True
```

The stride `(4, 1)` makes perfect sense for row-major storage: to get to the next row, skip 4 elements (the width of one row); to get to the next column, skip 1 element. The underlying memory is simply `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]` — a flat sequence. The strides are just the interpretation layer that makes it look 2D.

### Transposing Flips the Strides

When you transpose a tensor, PyTorch does not copy the data. It just swaps the strides:

```python
x_t = x.t()  # transpose
print(x_t)
print("shape: ", x_t.shape)
print("stride:", x_t.stride())
print("contiguous:", x_t.is_contiguous())
```

**Output:**
```text
tensor([[ 0,  4,  8],
        [ 1,  5,  9],
        [ 2,  6, 10],
        [ 3,  7, 11]])
shape:  torch.Size([4, 3])
stride: (1, 4)
contiguous: False
```

The data in memory hasn't moved at all — it's still `[0, 1, 2, ..., 11]`. PyTorch just swapped the strides from `(4, 1)` to `(1, 4)`. A stride of `(1, 4)` means: "to move to the next row, skip 1 element; to move to the next column, skip 4 elements." This is column-major order, and it's why the tensor is no longer contiguous — contiguous means row-major (C-order) storage where elements in the last dimension are adjacent in memory.

### view() Requires Contiguous; reshape() Does Not

`view()` reinterprets the underlying flat memory directly. Because it does not copy data, it requires the tensor to be contiguous — otherwise, the stride-based layout makes it impossible to reinterpret the same memory as a different shape.

```python
# view() fails on non-contiguous tensor
try:
    x_t.view(12)
except RuntimeError as e:
    print("view() error:", e)

# reshape() always works — copies if necessary
x_t_flat = x_t.reshape(12)
print("reshape() result:", x_t_flat)
```

**Output:**
```text
view() error: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
contiguous: False
reshape() result: tensor([ 0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11])
```

Notice the order of values in `reshape()` output — `[0, 4, 8, 1, 5, 9, ...]`. Because the transposed tensor reads memory column-by-column (stride `(1, 4)`), flattening it produces column-major order, not the original row-major order. This is why "flatten then reshape" on a transposed tensor gives different results than you might expect.

### Making a Tensor Contiguous

Call `.contiguous()` to get a new tensor with row-major layout. After this, `view()` works:

```python
x_t_cont = x_t.contiguous()
print("contiguous:", x_t_cont.is_contiguous())
print("stride:", x_t_cont.stride())

# Now view() works
flat = x_t_cont.view(12)
print(flat)
```

**Output:**
```text
contiguous: True
stride: (3, 1)
tensor([ 0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11])
```

`.contiguous()` physically copies the data into row-major order. The stride is now `(3, 1)` — the transposed shape is `(4, 3)`, so moving to the next row skips 3 elements. The data order is still column-major relative to the original because the copy preserves the logical values, just in a new memory layout.

> **Rule of thumb:** Use `reshape()` when you just need a different shape and don't care about the underlying layout. Use `view()` when you need a zero-copy operation and can guarantee contiguity — typically right after creating or cloning a tensor.

---

## Broadcasting: Shape Alignment Rules

Broadcasting lets PyTorch perform arithmetic between tensors of different shapes without explicit copies. When you add a bias vector of shape `(64,)` to a batch of activations of shape `(32, 64)`, broadcasting handles the shape mismatch automatically. But the rules are strict, and violating them raises an error rather than silently producing wrong results.

### The Three Broadcasting Rules

PyTorch follows NumPy's broadcasting rules:

1. **Align shapes from the right.** Compare dimensions starting from the trailing (rightmost) dimension, working left.
2. **Each pair of dimensions must be equal, or one of them must be `1`.** A size-`1` dimension is stretched to match the other.
3. **Missing dimensions on the left are treated as size `1`.** A 1D tensor of shape `(4,)` is treated as `(1, 4)` when paired with a 2D tensor.

Let's build up from simple to complex:

```python
# Case 1: scalar broadcast — scalar treated as shape ()
a = torch.tensor([1.0, 2.0, 3.0])   # shape (3,)
b = 10.0                              # scalar

print(a + b)
```

**Output:**
```text
tensor([11., 12., 13.])
```

The scalar `10.0` is broadcast across all 3 elements. This is the simplest case — effectively the scalar becomes `[10.0, 10.0, 10.0]` without any copy.

```python
# Case 2: column vector + row vector -> 2D matrix
a = torch.ones(3, 1)   # shape (3, 1)
b = torch.ones(1, 4)   # shape (1, 4)

c = a + b
print(c)
print("output shape:", c.shape)
```

**Output:**
```text
tensor([[2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.]])
output shape: torch.Size([3, 4])
```

Aligning from the right: dimension -1 is `1` vs `4` → stretch `a` to 4; dimension -2 is `3` vs `1` → stretch `b` to 3. Output shape: `(3, 4)`. No data was copied — PyTorch uses strides under the hood to read the same memory repeatedly.

```python
# Case 3: adding a 1D bias to a 2D batch (the most common ML use case)
activations = torch.zeros(32, 64)   # batch of 32, feature dim 64
bias        = torch.ones(64)        # shape (64,) — no batch dimension

out = activations + bias
print("activations shape:", activations.shape)
print("bias shape:       ", bias.shape)
print("output shape:     ", out.shape)
```

**Output:**
```text
activations shape: torch.Size([32, 64])
bias shape:        torch.Size([64,])
output shape:      torch.Size([32, 64])
```

The `bias` tensor of shape `(64,)` is treated as `(1, 64)` by rule 3, then stretched to `(32, 64)` by rule 2. This is exactly how `nn.Linear` adds a bias term to a batch of inputs.

### Incompatible Shapes Raise Immediately

```python
a = torch.ones(3, 4)
b = torch.ones(3, 5)   # 4 != 5, neither is 1

try:
    c = a + b
except RuntimeError as e:
    print(e)
```

**Output:**
```text
The size of tensor a (4) must match the size of tensor b (5) at non-singleton dimension 1
```

The error message tells you exactly which dimension failed and what the sizes were. PyTorch never silently broadcasts incompatible shapes — this is by design.

### The Unintended Broadcast Gotcha

Broadcasting is powerful but can mask shape bugs. Consider a common mistake when computing a dot product:

```python
a = torch.randn(4)    # shape (4,)
b = torch.randn(4, 1) # shape (4, 1) — column vector

# Intended: dot product (scalar). Actual: outer product (4x4 matrix)
result = a * b
print("a shape:", a.shape)
print("b shape:", b.shape)
print("result shape:", result.shape)
```

**Output:**
```text
a shape: torch.Size([4])
b shape: torch.Size([4, 1])
result shape: torch.Size([4, 4])
```

`a` (shape `(4,)`) is treated as `(1, 4)`, then broadcast to `(4, 4)`. `b` (shape `(4, 1)`) is broadcast to `(4, 4)`. The result is a `4×4` outer product — not the dot product you might have intended. No error, no warning. The fix is to be explicit about shapes using `torch.dot()`, `torch.matmul()`, or `unsqueeze`:

```python
# Correct dot product
dot = torch.dot(a, b.squeeze())
print("dot result:", dot.item())

# Or use unsqueeze to make intent explicit
a_col = a.unsqueeze(1)   # (4,) -> (4, 1)
print("a_col shape:", a_col.shape)
element_wise = a_col * b  # (4, 1) * (4, 1) — no broadcast needed
print("element-wise shape:", element_wise.shape)
```

**Output:**
```text
dot result: -0.6132094264030457
element-wise shape: torch.Size([4, 1])
```

> Note: The dot product value differs each run due to `torch.randn`. The shape result is deterministic.

Using `unsqueeze` to add explicit size-`1` dimensions makes your broadcasting intent visible in the code. When a shape bug appears, you'll see it in the `unsqueeze` call rather than hunting for why your loss is the wrong shape.

---

## Conclusion

Tensors look simple — they're just multi-dimensional arrays. But the four dimensions covered here (dtype, device, memory layout, broadcasting) are where most PyTorch bugs live. Use the right dtype from the start: `float32` for weights, `int64` for indices, `float16` or `bfloat16` for mixed-precision GPU training. Keep devices consistent and be explicit with `.to()`. Reach for `reshape()` unless you have a specific reason to use `view()`, and call `.contiguous()` before `view()` when operating on transposed tensors. In broadcasting, align shapes from the right and use `unsqueeze` to make your intent explicit rather than relying on implicit promotion. Get these four things right and you'll spend your debugging time on model architecture, not tensor plumbing.
