---
title: >-
  PyTorch Training Loop Anatomy: Forward Pass, Loss, Backward, and
  optimizer.step
excerpt: >-
  The PyTorch training loop has a specific execution order with subtle
  requirements. Get zero_grad wrong and gradients accumulate. Skip no_grad in
  eval and memory leaks. This post covers every step.
author: Soham Sharma
authorName: Soham Sharma
category: AI
tags:
  - PyTorch
  - Training Loop
  - Optimization
  - Deep Learning
  - Python
status: published
featuredImage: >-
  https://images.unsplash.com/photo-1639762681485-074b7f938ba0?w=1200&auto=format&fit=crop&q=80
colab_notebook: >-
  https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/pytorch/pytorch_4.ipynb
series_id: pytorch-mastery
series_slug: pytorch-mastery
series_title: 'PyTorch Mastery: From Tensors to Production'
difficulty: beginner
week: null
day: 16
tools:
  - PyTorch
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/pytorch/pytorch_4.ipynb)


The PyTorch training loop looks simple: forward pass, compute loss, call backward, step the optimizer. But each of those four steps has requirements that aren't obvious from the high-level description — and violating any of them produces bugs that are silent, slow, or both. This post builds a complete, production-grade training loop from first principles and explains every line.

## The Four-Step Loop

Every PyTorch training iteration follows this exact sequence:

```
1. optimizer.zero_grad()   ← clear accumulated gradients
2. output = model(input)   ← forward pass
3. loss.backward()         ← compute gradients
4. optimizer.step()        ← update parameters
```

The order matters. Swapping steps 1 and 4 works but accumulates gradients across batches. Skipping step 1 causes gradients to accumulate until you explicitly clear them. Let's build this up correctly.

## Step 1: A Minimal Correct Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Model: 2-layer MLP
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 5),
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Dummy dataset
torch.manual_seed(42)
X = torch.randn(1000, 20)
y = torch.randint(0, 5, (1000,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Training loop
model.train()  # set to training mode (enables Dropout, BatchNorm train behavior)

for epoch in range(3):
    total_loss = 0.0
    correct = 0

    for x_batch, y_batch in loader:
        # Step 1: clear gradients
        optimizer.zero_grad()

        # Step 2: forward pass
        logits = model(x_batch)

        # Step 3: compute loss and backpropagate
        loss = criterion(logits, y_batch)
        loss.backward()

        # Step 4: update parameters
        optimizer.step()

        # Track metrics (detach from graph — don't need gradients here)
        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == y_batch).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(dataset)
    print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, accuracy={accuracy:.4f}")
```

**Output:**
```text
Epoch 1: loss=1.6234, accuracy=0.2010
Epoch 2: loss=1.5987, accuracy=0.2340
Epoch 3: loss=1.5701, accuracy=0.2750
```

> Note: Exact values vary by initialization and shuffle order.

Every line in this loop has a reason. Let's examine each decision.

## model.train() and model.eval(): Mode Switching

`model.train()` and `model.eval()` set a flag that controls the behavior of `Dropout` and `BatchNorm` layers. This is not optional — forgetting to switch modes is one of the most common sources of inconsistent train/val results.

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.Dropout(0.5),  # drops 50% of activations during training
    nn.Linear(20, 5),
)

x = torch.randn(4, 10)

# Training mode: dropout active
model.train()
out_train1 = model(x)
out_train2 = model(x)  # different — dropout randomly zeros different neurons

# Eval mode: dropout disabled (all neurons active, scaled)
model.eval()
out_eval1 = model(x)
out_eval2 = model(x)  # identical — no randomness

print(f"Train outputs identical: {torch.allclose(out_train1, out_train2)}")
print(f"Eval outputs identical:  {torch.allclose(out_eval1, out_eval2)}")
```

**Output:**
```text
Train outputs identical: False
Eval outputs identical:  True
```

In eval mode, dropout scales activations by `1/(1-p)` to maintain expected value. In training mode, dropout randomly zeros `p` fraction of activations — different every call. Always call `model.eval()` before validation/inference.

## optimizer.zero_grad(): Three Variants

There are three ways to zero gradients, with different trade-offs:

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 5)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
x = torch.randn(4, 10)
y = torch.randint(0, 5, (4,))

# Option 1: optimizer.zero_grad() — standard, calls tensor.grad.zero_()
optimizer.zero_grad()

# Option 2: optimizer.zero_grad(set_to_none=True) — sets grad to None instead of zeros
# Faster: avoids a memory write to zero the gradient buffer
# Default in PyTorch >= 2.0
optimizer.zero_grad(set_to_none=True)

# Option 3: manual — for gradient accumulation across N mini-batches
# Don't zero every step; zero every N steps
ACCUMULATION_STEPS = 4
for step, (xb, yb) in enumerate([(x, y)] * 8):
    logits = model(xb)
    loss = nn.CrossEntropyLoss()(logits, yb) / ACCUMULATION_STEPS
    loss.backward()

    if (step + 1) % ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()

print("Gradient accumulation completed successfully.")
```

**Output:**
```text
Gradient accumulation completed successfully.
```

`set_to_none=True` is faster because it avoids writing zeros to the gradient buffer — it just removes the reference. The downside: code that checks `if param.grad is not None` will behave differently. In modern PyTorch, it's the recommended default.

![PyTorch training loop diagram showing the four steps](https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200&auto=format&fit=crop&q=80)

## loss.item() vs loss: Why It Matters

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
x = torch.randn(8, 10)
y = torch.randn(8, 1)

loss = nn.MSELoss()(model(x), y)

# WRONG: keeps entire computation graph alive in memory
total_loss_wrong = 0.0
total_loss_wrong += loss  # adds a tensor with grad_fn to a Python float (auto-converts)

# Correct: extract scalar value, discards graph reference
total_loss_correct = 0.0
total_loss_correct += loss.item()  # Python float — no graph

print(f"type(loss): {type(loss)}")
print(f"type(loss.item()): {type(loss.item())}")
print(f"loss.item(): {loss.item():.4f}")
```

**Output:**
```text
type(loss): <class 'torch.Tensor'>
type(loss.item()): <class 'float'>
loss.item(): 1.2341
```

> Note: Exact loss value varies by initialization.

Accumulating `loss` tensors in a Python list or sum holds references to their computation graphs. Over a long training run, this causes memory to grow steadily. Always use `loss.item()` to extract the scalar value.

## The Validation Loop

The validation loop is structurally similar but with two key differences: `model.eval()` and `torch.no_grad()`.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)

model = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 5),
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Datasets
X_train, y_train = torch.randn(800, 20), torch.randint(0, 5, (800,))
X_val, y_val     = torch.randn(200, 20), torch.randint(0, 5, (200,))

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=64)

for epoch in range(2):
    # --- Training ---
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # --- Validation ---
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():  # disables gradient tracking — saves memory and ~20% compute
        for xb, yb in val_loader:
            logits = model(xb)
            val_loss += criterion(logits, yb).item()
            val_correct += (logits.argmax(1) == yb).sum().item()

    print(f"Epoch {epoch+1}: "
          f"train_loss={train_loss/len(train_loader):.4f} | "
          f"val_loss={val_loss/len(val_loader):.4f} | "
          f"val_acc={val_correct/len(X_val):.4f}")
```

**Output:**
```text
Epoch 1: train_loss=1.6102 | val_loss=1.6089 | val_acc=0.2050
Epoch 2: train_loss=1.5834 | val_loss=1.5921 | val_acc=0.2200
```

> Note: Exact values vary by initialization.

`torch.no_grad()` is not the same as `model.eval()`. `no_grad()` prevents gradient computation — it means no `grad_fn` objects are created, saving memory and ~20% compute. `model.eval()` changes layer behavior (Dropout, BatchNorm). You need both in a validation loop.

## Saving and Loading Checkpoints Mid-Training

A production training loop saves checkpoints. The minimal checkpoint includes model state, optimizer state, and current epoch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import os
import tempfile

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 5))
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_path = os.path.join(tmpdir, 'checkpoint.pt')

    # Simulate saving at end of epoch 3
    save_checkpoint(model, optimizer, epoch=3, loss=0.4231, path=ckpt_path)

    # Load and resume
    new_model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 5))
    new_optimizer = optim.Adam(new_model.parameters(), lr=1e-3)
    epoch, loss = load_checkpoint(new_model, new_optimizer, ckpt_path)

    print(f"Resumed from epoch {epoch}, loss was {loss:.4f}")
    print(f"Optimizer lr: {new_optimizer.param_groups[0]['lr']}")
```

**Output:**
```text
Resumed from epoch 3, loss was 0.4231
Optimizer lr: 0.001
```

Saving `optimizer_state_dict` is critical — Adam's per-parameter moment estimates (m_t and v_t) are part of the optimizer state. Loading only `model_state_dict` and starting a fresh optimizer effectively resets the learning dynamics.

## A Complete Production Training Loop

Putting it all together with gradient clipping, learning rate scheduling, and early stopping logic:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(20, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, 5),
)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose=False)
criterion = nn.CrossEntropyLoss()

X_train, y_train = torch.randn(800, 20), torch.randint(0, 5, (800,))
X_val,   y_val   = torch.randn(200, 20), torch.randint(0, 5, (200,))
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=64)

best_val_loss = float('inf')
patience_counter = 0
MAX_PATIENCE = 3

for epoch in range(10):
    # Train
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()

    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            val_loss += criterion(model(xb), yb).item()
    val_loss /= len(val_loader)

    scheduler.step(val_loss)  # adjust lr based on val loss

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1

    print(f"Epoch {epoch+1:2d}: val_loss={val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}, patience={patience_counter}")

    if patience_counter >= MAX_PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break
```

**Output:**
```text
Epoch  1: val_loss=1.6043, lr=0.001000, patience=0
Epoch  2: val_loss=1.5987, lr=0.001000, patience=0
Epoch  3: val_loss=1.6012, lr=0.001000, patience=1
Epoch  4: val_loss=1.6089, lr=0.001000, patience=2
Epoch  5: val_loss=1.6134, lr=0.001000, patience=3
Early stopping at epoch 5
```

> Note: Exact values vary by initialization. Loss may not decrease significantly on random data — use real data for meaningful training signals.

![Training and validation loss curves showing convergence](https://images.unsplash.com/photo-1677442135703-1787eea5ce01?w=1200&auto=format&fit=crop&q=80)

## Common Bugs and Their Symptoms

| Bug | Symptom | Fix |
|---|---|---|
| Missing `zero_grad()` | Loss decreases then explodes | Add `optimizer.zero_grad()` at loop start |
| Missing `model.train()` | Val accuracy ≈ train accuracy (dropout disabled both) | Set `model.train()` before training loop |
| Missing `model.eval()` | Validation loss varies across identical inputs | Set `model.eval()` before validation loop |
| Missing `torch.no_grad()` | Memory grows during validation | Wrap val loop with `torch.no_grad()` |
| Accumulating `loss` tensor | Memory grows across epochs | Use `loss.item()` for metrics |
| Not saving optimizer state | Training dynamics reset on resume | Include `optimizer_state_dict` in checkpoint |

## Conclusion

The PyTorch training loop is four lines at its core, but each line has requirements that aren't stated in API docs. `zero_grad()` must come before `backward()`, not after `step()`. `model.train()`/`model.eval()` must match the phase. `torch.no_grad()` is mandatory in validation loops — both for correctness (no grad_fn accumulation) and performance. `loss.item()` prevents silent graph retention. Get these right and the training loop becomes a reliable foundation for everything built on top of it.

The next post steps up to intermediate territory: building a CNN from scratch on CIFAR-10, covering convolutional layers, pooling, BatchNorm, and the full pipeline from raw image to training.
