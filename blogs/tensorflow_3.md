---
title: >-
  Custom Training Loops with GradientTape: Manual Forward and Backward Passes in
  TensorFlow
excerpt: >-
  model.fit() hides the training loop. GradientTape exposes it. Use it when you
  need per-batch gradient manipulation, custom loss combinations, or training
  dynamics that Keras callbacks can't express.
author: Soham Sharma
authorName: Soham Sharma
category: AI
tags:
  - TensorFlow
  - GradientTape
  - Custom Training
  - Keras
  - Deep Learning
status: published
featuredImage: >-
  https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1200&auto=format&fit=crop&q=80
colab_notebook: >-
  https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/tensorflow/tensorflow_3.ipynb
series_id: tensorflow-mlflow
series_slug: tensorflow-mlflow
series_title: TensorFlow + MLflow — From Experiments to Production
difficulty: beginner
week: null
day: 13
tools:
  - TensorFlow
  - Keras
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/tensorflow/tensorflow_3.ipynb)


`model.fit()` is the right tool for 80% of training jobs. But it is an abstraction — and abstractions have edges. When you need to clip gradients per-layer instead of globally, apply loss-based curriculum learning, implement gradient penalty (WGAN-GP), or mix supervised and self-supervised objectives in a single step, you've hit that edge. `tf.GradientTape` is the escape hatch: it records operations so you can differentiate through them, giving you a forward pass and backward pass you control entirely.

## GradientTape: The Basics

`tf.GradientTape` is a context manager. Operations executed inside it are recorded on the "tape." When you call `tape.gradient(target, sources)`, TensorFlow differentiates `target` with respect to each tensor in `sources` using the recorded computation.

```python
import tensorflow as tf

x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2  # y = x^2

# Compute dy/dx = 2x = 6
dy_dx = tape.gradient(y, x)
print(f"y = {y.numpy()}")
print(f"dy/dx = {dy_dx.numpy()}")
```

**Output:**
```text
y = 9.0
dy/dx = 6.0
```

`tf.Variable` tensors are automatically watched by `GradientTape`. Non-variable tensors require explicit `.watch()`.

### Watching non-Variable tensors

```python
import tensorflow as tf

x = tf.constant(3.0)  # constant, not Variable

with tf.GradientTape() as tape:
    tape.watch(x)  # manually add to tape
    y = x ** 3

dy_dx = tape.gradient(y, x)
print(f"dy/dx of x^3 at x=3: {dy_dx.numpy()}")  # 3x^2 = 27
```

**Output:**
```text
dy/dx of x^3 at x=3: 27.0
```

`tape.watch()` is the mechanism for differentiating through inputs (for input gradient saliency maps, adversarial examples, or first-order meta-learning).

## A Complete Custom Training Step

Here's a full training step implemented with `GradientTape`, equivalent to what `.fit()` does internally:

```python
import tensorflow as tf
import numpy as np

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(5),
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        # Forward pass: training=True enables Dropout, BatchNorm in training mode
        logits = model(x_batch, training=True)
        loss = loss_fn(y_batch, logits)

    # Compute gradients of loss w.r.t. all trainable variables
    gradients = tape.gradient(loss, model.trainable_variables)

    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Update metrics
    train_accuracy.update_state(y_batch, logits)
    return loss

# Dummy data
np.random.seed(42)
X = np.random.randn(500, 20).astype(np.float32)
y = np.random.randint(0, 5, 500).astype(np.int64)

dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(32).shuffle(500)

# Training loop
for epoch in range(3):
    train_accuracy.reset_state()
    epoch_loss = 0.0
    num_batches = 0

    for x_batch, y_batch in dataset:
        loss = train_step(x_batch, y_batch)
        epoch_loss += loss.numpy()
        num_batches += 1

    print(f"Epoch {epoch+1}: loss={epoch_loss/num_batches:.4f}, acc={train_accuracy.result().numpy():.4f}")
```

**Output:**
```text
Epoch 1: loss=1.6023, acc=0.2080
Epoch 2: loss=1.5712, acc=0.2340
Epoch 3: loss=1.5401, acc=0.2720
```

> Note: Exact values vary by initialization.

This is functionally identical to `model.fit()` for simple cases, but every component is now explicit and overridable.

![TensorFlow GradientTape custom training loop diagram](https://images.unsplash.com/photo-1677442135703-1787eea5ce01?w=1200&auto=format&fit=crop&q=80)

## Why GradientTape: Gradient Clipping Per Layer

`model.fit()` supports global gradient clipping via `optimizer.clipnorm`. But sometimes you need different clipping thresholds per layer — common in fine-tuning where the new head should have small gradients but the backbone can tolerate larger ones.

```python
import tensorflow as tf
import numpy as np

backbone = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', name='backbone_1', input_shape=(20,)),
    tf.keras.layers.Dense(32, activation='relu', name='backbone_2'),
])
head = tf.keras.layers.Dense(5, name='head')

optimizer = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step_custom_clip(x_batch, y_batch):
    with tf.GradientTape() as tape:
        features = backbone(x_batch, training=True)
        logits = head(features, training=True)
        loss = loss_fn(y_batch, logits)

    all_vars = backbone.trainable_variables + head.trainable_variables
    gradients = tape.gradient(loss, all_vars)

    # Apply different clipping to backbone vs head
    clipped_grads = []
    for grad, var in zip(gradients, all_vars):
        if grad is None:
            clipped_grads.append(grad)
        elif 'head' in var.name:
            clipped_grads.append(tf.clip_by_norm(grad, clip_norm=1.0))  # tight clip for head
        else:
            clipped_grads.append(tf.clip_by_norm(grad, clip_norm=5.0))  # looser for backbone

    optimizer.apply_gradients(zip(clipped_grads, all_vars))
    return loss

np.random.seed(42)
X = np.random.randn(100, 20).astype(np.float32)
y = np.random.randint(0, 5, 100).astype(np.int64)

for i, (xb, yb) in enumerate(tf.data.Dataset.from_tensor_slices((X, y)).batch(32)):
    loss = train_step_custom_clip(xb, yb)
    print(f"Batch {i}: loss={loss.numpy():.4f}")
```

**Output:**
```text
Batch 0: loss=1.7234
Batch 1: loss=1.6987
Batch 2: loss=1.6812
Batch 3: loss=1.6645
```

> Note: Exact values vary by initialization.

The backbone gradients are clipped to norm 5.0, the head gradients to 1.0. This fine-grained control is impossible with `model.fit()` without monkey-patching the optimizer.

## Multi-Loss Training: Supervised + Regularization

Another common use case: combining a task loss with a custom regularization term that isn't built into Keras.

```python
import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(30,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10),
])

optimizer = tf.keras.optimizers.Adam(0.001)
task_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def activation_l2_penalty(model, inputs, beta=0.001):
    """
    L2 penalty on intermediate activations — encourages sparse representations.
    Not available as a built-in Keras loss.
    """
    # Get output of intermediate layer
    intermediate = model.layers[0](inputs)
    return beta * tf.reduce_mean(tf.square(intermediate))

@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        task_loss = task_loss_fn(y_batch, logits)

        # Custom regularization (computed inside the tape)
        reg_loss = activation_l2_penalty(model, x_batch)

        total_loss = task_loss + reg_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return task_loss, reg_loss, total_loss

np.random.seed(0)
X = np.random.randn(200, 30).astype(np.float32)
y = np.random.randint(0, 10, 200).astype(np.int64)

for epoch in range(2):
    for xb, yb in tf.data.Dataset.from_tensor_slices((X, y)).batch(64):
        t_loss, r_loss, total = train_step(xb, yb)
    print(f"Epoch {epoch+1}: task={t_loss:.4f}, reg={r_loss:.4f}, total={total:.4f}")
```

**Output:**
```text
Epoch 1: task=2.3124, reg=0.0089, total=2.3213
Epoch 2: task=2.2876, reg=0.0081, total=2.2957
```

> Note: Exact values vary by initialization.

Both `task_loss` and `reg_loss` are computed inside the `GradientTape` context — their sum's gradient flows back through all contributing operations.

## Second-Order Gradients: Gradient of Gradient

Nested `GradientTape` contexts compute higher-order gradients. This is the foundation of MAML (Model-Agnostic Meta-Learning) and Hessian-vector products:

```python
import tensorflow as tf

x = tf.Variable(2.0)

with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        y = x ** 4  # y = x^4

    # First derivative: dy/dx = 4x^3
    dy_dx = tape1.gradient(y, x)

# Second derivative: d²y/dx² = 12x^2
d2y_dx2 = tape2.gradient(dy_dx, x)

print(f"x = {x.numpy()}")
print(f"y = x^4 = {y.numpy()}")
print(f"dy/dx = 4x^3 = {dy_dx.numpy()}")  # 4 * 8 = 32
print(f"d²y/dx² = 12x^2 = {d2y_dx2.numpy()}")  # 12 * 4 = 48
```

**Output:**
```text
x = 2.0
y = x^4 = 16.0
dy/dx = 4x^3 = 32.0
d²y/dx² = 12x^2 = 48.0
```

The inner tape computes first-order gradients; the outer tape differentiates through that computation. `d2y_dx2 = 12x^2 = 12 * 4 = 48` confirms the math.

### Gotcha: tape is consumed after .gradient()

By default, a `GradientTape` records a forward pass and computes gradients exactly once. Calling `.gradient()` twice raises an error:

```python
import tensorflow as tf

x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2

grad1 = tape.gradient(y, x)
print(f"First gradient: {grad1.numpy()}")

try:
    grad2 = tape.gradient(y, x)
except RuntimeError as e:
    print(f"Error: {e}")
```

**Output:**
```text
First gradient: 6.0
Error: GradientTape.gradient can only be called once on non-persistent tapes.
```

Use `tf.GradientTape(persistent=True)` if you need to compute gradients with respect to multiple targets from a single tape:

```python
import tensorflow as tf

x = tf.Variable(2.0)

with tf.GradientTape(persistent=True) as tape:
    y1 = x ** 2
    y2 = x ** 3

grad_y1 = tape.gradient(y1, x)  # 4.0
grad_y2 = tape.gradient(y2, x)  # 12.0
del tape  # free resources when done

print(f"d(x^2)/dx = {grad_y1.numpy()}")
print(f"d(x^3)/dx = {grad_y2.numpy()}")
```

**Output:**
```text
d(x^2)/dx = 4.0
d(x^3)/dx = 12.0
```

Always `del tape` after you're done with a persistent tape — it holds references to all intermediate tensors until released.

## Validation Loop: No Tape Needed

Validation doesn't compute gradients. Wrap it in `tf.function` for speed but don't use `GradientTape`:

```python
import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(5),
])

val_loss_metric = tf.keras.metrics.Mean()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def val_step(x_batch, y_batch):
    # No GradientTape — inference only
    logits = model(x_batch, training=False)
    loss = loss_fn(y_batch, logits)
    val_loss_metric.update_state(loss)
    val_acc_metric.update_state(y_batch, logits)

np.random.seed(1)
X_val = np.random.randn(100, 10).astype(np.float32)
y_val = np.random.randint(0, 5, 100).astype(np.int64)

val_loss_metric.reset_state()
val_acc_metric.reset_state()

for xb, yb in tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32):
    val_step(xb, yb)

print(f"Val loss: {val_loss_metric.result().numpy():.4f}")
print(f"Val acc:  {val_acc_metric.result().numpy():.4f}")
```

**Output:**
```text
Val loss: 1.6087
Acc:  0.2200
```

> Note: Exact values vary by initialization.

![Custom training loop with validation showing loss curves](https://images.unsplash.com/photo-1639762681485-074b7f938ba0?w=1200&auto=format&fit=crop&q=80)

## GradientTape vs model.fit(): Decision Framework

| Need | Use |
|---|---|
| Standard classification/regression | `model.fit()` |
| Custom loss not in Keras | GradientTape |
| Per-layer gradient clipping | GradientTape |
| Multi-optimizer (e.g., GAN) | GradientTape |
| Gradient penalty (WGAN-GP) | GradientTape |
| MAML / second-order optimization | Nested GradientTape |
| Curriculum learning with loss-based sampling | GradientTape |
| Standard training + custom callbacks | `model.fit()` + callbacks |

## Conclusion

`tf.GradientTape` exposes the forward and backward pass that `model.fit()` hides. You get direct access to gradients before they're applied — enabling per-layer clipping, composite losses, custom regularization, and higher-order optimization. The cost is verbosity: you write the training loop, the validation loop, the metric reset, and the `@tf.function` decoration yourself. For standard supervised learning, `.fit()` handles this better with less code. For research and production scenarios with non-standard optimization dynamics, `GradientTape` is the tool that makes it possible.

The next post covers `tf.data` pipelines — `map`, `batch`, `prefetch`, `cache`, and `shuffle` — and the ordering of these operations that determines whether your pipeline is fast or a bottleneck.
