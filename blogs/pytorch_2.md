---
title: >-
  PyTorch Autograd Internals: Computation Graphs, retain_graph, grad_fn Chain,
  and detach
excerpt: >-
  Autograd is not magic — it's a directed acyclic graph of Function nodes.
  Understand how gradients flow, when retain_graph matters, and how detach
  prevents gradient leaks.
author: Soham Sharma
authorName: Soham Sharma
category: PyTorch
tags:
  - PyTorch
  - Autograd
  - Backpropagation
  - Deep Learning
  - Gradients
status: published
featuredImage: >-
  https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1200&auto=format&fit=crop&q=80
colab_notebook: >-
  https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/pytorch/pytorch_2.ipynb
series_id: pytorch-mastery
series_slug: pytorch-mastery
series_title: 'PyTorch Mastery: From Tensors to Production'
difficulty: beginner
week: null
day: 6
tools:
  - PyTorch
---

<a href="https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/pytorch/pytorch_2.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="height:28px;margin-bottom:1rem;" /></a>




Every time you call `loss.backward()`, PyTorch traverses a graph of operations you built during the forward pass and computes gradients using the chain rule. Most engineers treat this as a black box and wonder why `backward()` fails with "graph freed", why gradients are `None`, or why memory grows during training. These bugs disappear once you understand the computation graph. This post builds that understanding from the ground up.

## The Computation Graph: A DAG of Function Nodes

When you perform operations on tensors with `requires_grad=True`, PyTorch creates a **computation graph** — a Directed Acyclic Graph (DAG) where:
- **Leaf nodes** are tensors you created directly (weights, inputs with `requires_grad=True`)
- **Interior nodes** are `Function` objects representing operations (e.g., `AddBackward0`, `MulBackward0`, `MmBackward0`)
- **Edges** record which tensors each function consumed

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

z = x * y       # MulBackward0
w = z + x       # AddBackward0
out = w ** 2    # PowBackward0

print(f"out.grad_fn:           {out.grad_fn}")
print(f"out.grad_fn.next_functions: {out.grad_fn.next_functions}")
```

**Output:**
```text
out.grad_fn:           <PowBackward0 object at 0x7f3a1c2b4d30>
out.grad_fn.next_functions: ((<AddBackward0 object at 0x7f3a1c2b4df0>, 0),)
```

`out.grad_fn` is the `PowBackward0` node — the last operation that produced `out`. `next_functions` points to its parent in the graph: `AddBackward0`. Following the chain leads back to the leaf tensors `x` and `y`.

### Walking the full graph

```python
import torch

def walk_graph(tensor, depth=0):
    if tensor.grad_fn is None:
        print("  " * depth + f"Leaf: {tensor}")
        return
    print("  " * depth + f"{tensor.grad_fn.__class__.__name__}")
    for parent, _ in tensor.grad_fn.next_functions:
        if parent is not None:
            walk_graph_fn(parent, depth + 1)

def walk_graph_fn(fn, depth=0):
    print("  " * depth + f"{fn.__class__.__name__}")
    for parent, _ in fn.next_functions:
        if parent is not None:
            walk_graph_fn(parent, depth + 1)

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = x * y
w = z + x
out = w ** 2

walk_graph(out)
```

**Output:**
```text
PowBackward0
  AddBackward0
    MulBackward0
      AccumulateGrad
      AccumulateGrad
    AccumulateGrad
```

`AccumulateGrad` is the leaf node type for tensors with `requires_grad=True`. When `backward()` reaches an `AccumulateGrad` node, it writes the gradient into `tensor.grad`. The tree shape reflects the math: `out = (x*y + x)^2`, so backward traces through `Pow → Add → (Mul → x, Mul → y), x`.

## How backward() Works

Calling `out.backward()` triggers **reverse-mode automatic differentiation**:

1. Start at the root (`out.grad_fn`) with gradient `d(out)/d(out) = 1`
2. Call each `Function`'s `backward()` method, which computes the local Jacobian-vector product
3. Pass the result to each parent node
4. Accumulate gradients at leaf nodes into `.grad`

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# out = (x*y + x)^2 = (2*3 + 2)^2 = 64
z = x * y
w = z + x
out = w ** 2

print(f"out = {out.item()}")

out.backward()

# d(out)/dx = 2*(x*y+x) * (y+1) = 2*8*4 = 64
# d(out)/dy = 2*(x*y+x) * x    = 2*8*2 = 32
print(f"x.grad = {x.grad}")
print(f"y.grad = {y.grad}")
```

**Output:**
```text
out = 64.0
x.grad = 64.0
y.grad = 32.0
```

Let's verify by hand: `out = (xy + x)^2`. Using the chain rule: `∂out/∂x = 2(xy+x)(y+1) = 2(8)(4) = 64`. `∂out/∂y = 2(xy+x)(x) = 2(8)(2) = 32`. Both match exactly.

![Computation graph visualization showing forward and backward passes](https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200&auto=format&fit=crop&q=80)

## Graph Freeing: Why backward() Can Only Be Called Once

After `backward()` completes, PyTorch **frees the intermediate buffers** stored in the graph nodes. This is a memory optimization — those intermediate activations are only needed for gradient computation. Once gradients are computed, they're discarded.

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
out = x ** 3

out.backward()  # first call — works fine
print(f"x.grad = {x.grad}")

try:
    out.backward()  # second call — graph is already freed
except RuntimeError as e:
    print(f"RuntimeError: {e}")
```

**Output:**
```text
x.grad = 12.0
RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.
```

The error message is exactly right: use `retain_graph=True` if you need to call `backward()` multiple times on the same graph.

## retain_graph=True: When You Actually Need It

`retain_graph=True` prevents PyTorch from freeing the computation graph after `backward()`. The two real use cases:

**1. Multi-loss training with shared forward pass**

```python
import torch
import torch.nn as nn

# Shared encoder
encoder = nn.Linear(10, 5)
x = torch.randn(4, 10)
features = encoder(x)

# Two task heads
head_a = nn.Linear(5, 2)
head_b = nn.Linear(5, 3)

out_a = head_a(features)
out_b = head_b(features)

# Two separate losses
loss_a = out_a.mean()
loss_b = out_b.mean()

# Must retain_graph for first backward because features is used by both
loss_a.backward(retain_graph=True)
loss_b.backward()  # no retain needed for the last backward

print(f"encoder bias grad: {encoder.bias.grad}")
```

**Output:**
```text
encoder bias grad: tensor([-0.1342,  0.0891,  0.0234, -0.0567,  0.0123])
```

> Note: Exact gradient values vary by random initialization.

Without `retain_graph=True` on `loss_a.backward()`, the graph through `features` would be freed, and `loss_b.backward()` would fail.

**2. Higher-order gradients (gradient of gradient)**

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 3  # dy/dx = 3x^2 = 27

# First backward — retain graph for second-order computation
grad_x = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"dy/dx = {grad_x.item()}")  # 27

# Second backward — computes d²y/dx²
grad_x2 = torch.autograd.grad(grad_x, x)[0]
print(f"d²y/dx² = {grad_x2.item()}")  # 6x = 18
```

**Output:**
```text
dy/dx = 27.0
d²y/dx² = 18.0
```

`create_graph=True` is the `autograd.grad` equivalent of `retain_graph=True` — it keeps the graph alive so you can differentiate through the gradient itself.

### Gotcha: retain_graph causes memory leaks

`retain_graph=True` in a training loop causes the graph to accumulate across iterations because old graphs are never freed. This is a very common cause of memory growth during training:

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
x = torch.randn(32, 10)
y = torch.randn(32, 1)

# WRONG — retain_graph=True in a loop
for i in range(5):
    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    loss.backward(retain_graph=True)  # graph accumulates!
    print(f"Step {i}: loss={loss.item():.4f}, graph retained")
```

**Output:**
```text
Step 0: loss=1.0234, graph retained
Step 1: loss=1.0234, graph retained
Step 2: loss=1.0234, graph retained
Step 3: loss=1.0234, graph retained
Step 4: loss=1.0234, graph retained
```

> Note: Loss values vary by initialization. Notice they don't decrease — gradients accumulate without optimizer step being called, but the graph memory grows with each iteration.

Unless you have a specific reason to call `backward()` multiple times (multi-loss or higher-order gradients), never use `retain_graph=True` in a training loop.

## detach(): Cutting the Graph

`detach()` returns a tensor that shares the same data but is excluded from the computation graph — it has no `grad_fn` and `requires_grad=False`.

```python
import torch
import torch.nn as nn

encoder = nn.Linear(10, 5)
decoder = nn.Linear(5, 10)

x = torch.randn(4, 10)
encoded = encoder(x)

# Detach: decoder receives the values but not the gradient
encoded_detached = encoded.detach()

reconstructed = decoder(encoded_detached)
loss = ((reconstructed - x) ** 2).mean()
loss.backward()

print(f"decoder bias grad: {decoder.bias.grad is not None}")
print(f"encoder bias grad: {encoder.bias.grad}")   # None — gradient was cut
```

**Output:**
```text
decoder bias grad: True
encoder bias grad: None
```

The gradient does not flow back through `encoded_detached` to `encoder`. This is the standard pattern for **stop-gradient** operations — used in GANs (stop gradient through discriminator when training generator), self-supervised learning (stop gradient through the momentum encoder), and reinforcement learning (stop gradient through target networks).

### detach() vs no_grad()

These are not the same:

| | `detach()` | `torch.no_grad()` |
|---|---|---|
| Scope | Single tensor | Code block |
| Memory | Creates new tensor view | No allocation |
| Use case | Cut gradient for one tensor in a live graph | Inference / evaluation loop |

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
x = torch.randn(4, 10)

# no_grad: no gradient tracking for the entire block
with torch.no_grad():
    out = model(x)
    print(f"out.requires_grad (no_grad): {out.requires_grad}")
    print(f"out.grad_fn: {out.grad_fn}")

# detach: creates a view with gradient tracking removed
out_live = model(x)
out_detached = out_live.detach()
print(f"out_live.requires_grad: {out_live.requires_grad}")
print(f"out_detached.requires_grad: {out_detached.requires_grad}")
```

**Output:**
```text
out.requires_grad (no_grad): False
out.grad_fn: None
out_live.requires_grad: True
out_detached.requires_grad: False
```

Use `torch.no_grad()` for inference loops — it's more efficient because no `grad_fn` objects are created at all. Use `detach()` when you need to cut the gradient for a specific tensor while keeping the graph alive for other tensors.

## Gradient Accumulation in leaf .grad

Leaf tensor gradients accumulate across `backward()` calls. This is intentional — it enables **gradient accumulation** (simulating large batch sizes) — but it also bites engineers who forget to zero gradients.

```python
import torch

x = torch.tensor(1.0, requires_grad=True)

for i in range(3):
    y = x * (i + 1)
    y.backward()
    print(f"After step {i}: x.grad = {x.grad}")
```

**Output:**
```text
After step 0: x.grad = 1.0
After step 1: x.grad = 3.0
After step 2: x.grad = 6.0
```

`x.grad` accumulates: 1 → 1+2 → 1+2+3 = 6. In a training loop, this means calling `optimizer.zero_grad()` (or `model.zero_grad()`) before each forward pass is not optional — it's required to prevent gradient accumulation across batches.

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(5, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(8, 5)
y = torch.randn(8, 1)

for step in range(3):
    optimizer.zero_grad()           # ← clear accumulated gradients
    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    print(f"Step {step}: loss={loss.item():.4f}, bias.grad={model.bias.grad.item():.4f}")
```

**Output:**
```text
Step 0: loss=1.3421, bias.grad=-0.2134
Step 1: loss=1.2987, bias.grad=-0.1876
Step 2: loss=1.2603, bias.grad=-0.1654
```

> Note: Exact values vary by initialization.

![Neural network backpropagation gradient flow diagram](https://images.unsplash.com/photo-1639762681485-074b7f938ba0?w=1200&auto=format&fit=crop&q=80)

## Practical Autograd Checklist

Before shipping a training loop, verify:

- `optimizer.zero_grad()` is called before each `loss.backward()`
- `retain_graph=True` is only present where genuinely needed (multi-loss or higher-order)
- `torch.no_grad()` wraps all validation/inference code
- `detach()` is used for stop-gradient operations (GAN discriminator, target networks)
- No numpy conversion on GPU tensors without `.detach().cpu()` first

## Conclusion

The computation graph is not abstract theory — it directly explains every autograd behavior you encounter in practice. Graphs are freed after `backward()` by design; `retain_graph=True` is the escape hatch but creates memory leaks if misused. `detach()` surgically cuts gradient flow for one tensor. `.grad` accumulates, so zero it. Understanding these mechanics means you stop guessing at autograd errors and start predicting them.

The next post covers `Dataset` and `DataLoader` — how PyTorch manages data pipelines, why `num_workers` matters, and how to write a correct `collate_fn`.
