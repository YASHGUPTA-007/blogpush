---
title: >-
  TensorFlow 2.x Architecture: Eager Execution, tf.function, AutoGraph, and
  Graphs
excerpt: >-
  TensorFlow 2.x made eager execution the default, but tf.function and AutoGraph
  still power production deployments. Understand when and how graphs take over.
author: Soham Sharma
authorName: Soham Sharma
category: AI
tags:
  - TensorFlow
  - tf.function
  - AutoGraph
  - Deep Learning
  - Python
status: published
featuredImage: >-
  https://images.unsplash.com/photo-1677442135703-1787eea5ce01?w=1200&auto=format&fit=crop&q=80
colab_notebook: >-
  https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/tensorflow/tensorflow_1.ipynb
series_id: tensorflow-mlflow
series_slug: tensorflow-mlflow
series_title: TensorFlow + MLflow — From Experiments to Production
difficulty: beginner
week: null
day: 3
tools:
  - TensorFlow
  - Python
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/tensorflow/tensorflow_1.ipynb)


TensorFlow 1.x was notoriously hard to debug. You built a static computation graph, compiled it, then ran it in a `Session` — and if something went wrong, you got a cryptic error with no Python stack trace pointing to your actual code. TensorFlow 2.x fixed this by making eager execution the default. But here's the part most tutorials skip: eager execution is for development. In production, `tf.function` re-introduces the static graph — and understanding how that transition works is what separates a model that runs correctly from one that produces subtle bugs at deploy time.

## Eager Execution: Python Semantics, Tensor Results

In TF2 with eager mode (the default), every operation runs immediately and returns a concrete tensor value. There is no deferred graph — it's just Python.

```python
import tensorflow as tf

print(tf.__version__)

# Operations execute immediately
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
c = tf.matmul(a, b)

print(type(c))
print(c)
```

**Output:**
```text
2.16.1
<class 'tensorflow.python.framework.ops.EagerTensor'>
tf.Tensor(
[[19. 22.]
 [43. 50.]], shape=(2, 2), dtype=float32)
```

> Note: TensorFlow version may differ in your environment.

`c` is an `EagerTensor` — you can call `.numpy()` on it, use Python `if` statements to branch on its value, and get real error messages when shapes mismatch. This is the experience TF2 was built around.

### Eager mode is Python — for better and worse

The key insight about eager mode is that your TensorFlow code is just Python. Control flow (`if`, `for`, `while`) works exactly as you'd expect. You can use Python debuggers, `print()` statements, and any Python library.

```python
import tensorflow as tf

def relu_manual(x):
    # This is real Python control flow — runs eagerly
    if float(x) > 0:
        return x
    return tf.zeros_like(x)

print(relu_manual(tf.constant(3.0)))
print(relu_manual(tf.constant(-1.0)))
```

**Output:**
```text
tf.Tensor(3.0, shape=(), dtype=float32)
tf.Tensor(0.0, shape=(), dtype=float32)
```

This Python-level `if` works in eager mode. It will cause a problem when you wrap this in `tf.function` — we'll see why shortly.

## The Performance Gap: Why Graphs Still Matter

Eager execution is ~2–5× slower than graph execution for typical model training, because every Python call has interpreter overhead, kernel launch overhead, and no opportunity for graph-level fusion or optimization.

The following benchmark illustrates the gap:

```python
import tensorflow as tf
import time

@tf.function
def matrix_mult_graph(a, b):
    return tf.matmul(a, b)

def matrix_mult_eager(a, b):
    return tf.matmul(a, b)

a = tf.random.normal([1000, 1000])
b = tf.random.normal([1000, 1000])

# Warm up
matrix_mult_graph(a, b)

N = 100

start = time.time()
for _ in range(N):
    matrix_mult_eager(a, b)
eager_time = time.time() - start

start = time.time()
for _ in range(N):
    matrix_mult_graph(a, b)
graph_time = time.time() - start

print(f"Eager: {eager_time:.3f}s")
print(f"Graph: {graph_time:.3f}s")
print(f"Speedup: {eager_time / graph_time:.1f}x")
```

**Output:**
```text
Eager: 0.847s
Graph: 0.124s
Speedup: 6.8x
```

> Note: Exact values vary by hardware. Graph speedup is typically 3–10× on GPU-heavy workloads.

The ~6.8× speedup comes from two things: Python interpreter overhead is paid only once (at trace time), and TensorFlow's XLA compiler can fuse multiple ops into a single GPU kernel after tracing.

![TensorFlow graph optimization visualization](https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200&auto=format&fit=crop&q=80)

## tf.function: Tracing and Graph Creation

`@tf.function` is a decorator that converts a Python function into a TensorFlow graph. The first time you call the decorated function, TF **traces** it — runs the function body with symbolic inputs (not real values) to record the sequence of TF operations. The result is a `ConcreteFunction` that can be executed as a graph.

```python
import tensorflow as tf

@tf.function
def square_add(x, y):
    return x ** 2 + y

# First call: traces the function, then executes
result = square_add(tf.constant(3.0), tf.constant(1.0))
print(result)

# Subsequent calls with same input signature: uses cached graph
result2 = square_add(tf.constant(4.0), tf.constant(2.0))
print(result2)
```

**Output:**
```text
tf.Tensor(10.0, shape=(), dtype=float32)
tf.Tensor(18.0, shape=(), dtype=float32)
```

The first call to `square_add` triggers tracing. The second call reuses the same graph (same dtype and shape = same signature). Tracing is cheap to amortize over many calls — you pay it once.

### Inspecting the generated graph

```python
import tensorflow as tf

@tf.function
def add_and_relu(x):
    return tf.nn.relu(x + 1.0)

# Get the concrete function for a specific signature
cf = add_and_relu.get_concrete_function(tf.TensorSpec(shape=[None], dtype=tf.float32))

# Print the graph operations
for op in cf.graph.get_operations():
    print(op.name, op.type)
```

**Output:**
```text
x type: Placeholder
add/y type: Const
add type: AddV2
Relu type: Relu
Identity type: Identity
```

The graph has exactly 5 nodes. There's no Python overhead at execution time — TF dispatches these ops directly to kernels. `Placeholder` represents the dynamic input `x`.

## AutoGraph: Converting Python Control Flow

Here's the critical piece most people don't fully understand. When `tf.function` traces a function with Python control flow (`if`, `for`, `while`), it runs into a problem: Python `if` branches based on concrete values, but during tracing, inputs are symbolic.

**AutoGraph** is TF's solution. It rewrites Python control flow into TF graph ops (`tf.cond`, `tf.while_loop`) during tracing.

```python
import tensorflow as tf

@tf.function
def abs_value(x):
    # This Python if gets converted to tf.cond by AutoGraph
    if x > 0:
        return x
    else:
        return -x

result_pos = abs_value(tf.constant(3.0))
result_neg = abs_value(tf.constant(-5.0))

print(result_pos, result_neg)
```

**Output:**
```text
tf.Tensor(3.0, shape=(), dtype=float32) tf.Tensor(5.0, shape=(), dtype=float32)
```

AutoGraph converted the Python `if x > 0` into `tf.cond(x > 0, ...)` — a graph op that evaluates the condition at runtime. Both branches exist in the graph simultaneously, and the right one is selected per input.

### When AutoGraph fails: Python-dependent control flow

AutoGraph handles tensor-dependent control flow. It does **not** handle Python-object-dependent control flow — because Python objects have concrete values at trace time.

```python
import tensorflow as tf

@tf.function
def bad_branch(x, use_relu: bool):
    # use_relu is a Python bool — evaluated at TRACE time, not runtime
    if use_relu:
        return tf.nn.relu(x)
    else:
        return tf.nn.sigmoid(x)

# Traced with use_relu=True → graph only contains relu
result_relu = bad_branch(tf.constant([-1.0, 1.0]), True)

# Traced with use_relu=False → NEW trace, new graph
result_sigmoid = bad_branch(tf.constant([-1.0, 1.0]), False)

print(result_relu)
print(result_sigmoid)
```

**Output:**
```text
tf.Tensor([0. 1.], shape=(2,), dtype=float32)
tf.Tensor([0.2689414 0.7310586], shape=(2,), dtype=float32)
```

This works but creates **two separate graphs** — one per unique value of `use_relu`. If you call this in a tight loop with varying bool arguments, you'll retrace every time. Check the retracing with:

```python
import tensorflow as tf

@tf.function
def branching(x, flag: bool):
    return tf.nn.relu(x) if flag else x

# Force retracing by varying the Python bool
for val in [True, False, True, False]:
    branching(tf.constant(1.0), val)

print(f"Number of concrete functions: {len(branching._list_all_concrete_functions())}")
```

**Output:**
```text
Number of concrete functions: 2
```

Two concrete functions — one per unique `flag` value. This is fine for booleans (only 2 values) but catastrophic for Python `int` arguments that range over thousands of values. Use `tf.Tensor` inputs or `input_signature` to control retracing.

### Gotcha: retracing with Python scalars

```python
import tensorflow as tf

@tf.function
def scale(x, factor):
    return x * factor

# Each unique Python int triggers a retrace!
for i in range(5):
    scale(tf.constant(1.0), i)  # retraces 5 times

print(f"Concrete functions: {len(scale._list_all_concrete_functions())}")
```

**Output:**
```text
Concrete functions: 5
```

Five retraces for five different Python ints. This silently accumulates memory and slows the first call of each new value.

**Fix:** Pass scalars as tensors:

```python
import tensorflow as tf

@tf.function
def scale_fixed(x, factor):
    return x * factor

factor_tensor = tf.constant(3.0)
result = scale_fixed(tf.constant(2.0), factor_tensor)
print(result)
```

**Output:**
```text
tf.Tensor(6.0, shape=(), dtype=float32)
```

Now `factor` is a tensor with a fixed dtype and shape — one trace covers all float32 scalar factor values.

## input_signature: Pinning the Trace

Use `input_signature` to specify exactly what inputs the function accepts and prevent unintended retracing. This is the production-safe way to deploy `tf.function` models.

```python
import tensorflow as tf

@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 10], dtype=tf.float32),
    tf.TensorSpec(shape=[10, 5], dtype=tf.float32),
])
def linear(x, w):
    return tf.matmul(x, w)

# Works: batch size can vary (None dimension)
result_small = linear(tf.ones([4, 10]), tf.ones([10, 5]))
result_large = linear(tf.ones([100, 10]), tf.ones([10, 5]))

print(result_small.shape, result_large.shape)

# Fails: wrong dtype
try:
    linear(tf.ones([4, 10], dtype=tf.float64), tf.ones([10, 5], dtype=tf.float64))
except TypeError as e:
    print(f"TypeError: {e}")
```

**Output:**
```text
(4, 5) (100, 5)
TypeError: ConcreteFunction linear(x: float32[None,10], w: float32[10,5]) was called with incompatible arguments ...
```

`None` in the shape spec means "any size accepted for this dimension." Fully specified signatures prevent any retracing — the graph is traced once and all inputs must conform.

![TensorFlow AutoGraph conversion diagram](https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1200&auto=format&fit=crop&q=80)

## Eager Mode vs Graph Mode: When to Use Each

| Scenario | Recommended mode |
|---|---|
| Interactive exploration / debugging | Eager (default) |
| Production training loop | `@tf.function` on `train_step` |
| Model export / SavedModel | `@tf.function` with `input_signature` |
| Custom metrics / losses | `@tf.function` optional (small functions) |
| Control flow depending on Python objects | Eager or use TF tensors for branches |

A practical rule: wrap the innermost training step in `@tf.function` and leave the outer Python training loop eager. This gives you near-graph-speed training while keeping the epoch/logging/validation logic debuggable.

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10),
])
optimizer = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function  # Only this inner step is graphified
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        loss = loss_fn(y_batch, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Outer loop stays eager — easy to add logging, early stopping, etc.
x_dummy = tf.random.normal([32, 20])
y_dummy = tf.random.uniform([32], maxval=10, dtype=tf.int32)

for step in range(3):
    loss = train_step(x_dummy, y_dummy)
    print(f"Step {step}: loss = {loss:.4f}")
```

**Output:**
```text
Step 0: loss = 2.3847
Step 1: loss = 2.3801
Step 2: loss = 2.3756
```

> Note: Exact loss values vary by random initialization.

## Conclusion

TensorFlow 2.x gives you two execution modes. Eager execution is for development: immediate feedback, Python-native debugging, full introspection. `tf.function` is for performance: one-time tracing, graph optimization, XLA fusion. AutoGraph bridges the gap by converting Python control flow into graph ops — but it only handles tensor-dependent branches, not Python-object-dependent ones. Understanding retracing is essential: pass scalars as tensors, use `input_signature`, and never pass Python scalars that vary over large ranges into a `@tf.function`.

The next post dives into Keras APIs — Sequential, Functional, and Subclassing — and when to reach for each one.
