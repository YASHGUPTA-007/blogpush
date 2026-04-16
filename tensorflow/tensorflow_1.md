---
title: "TensorFlow 2.x Architecture: Eager Execution, tf.function, AutoGraph, and Graphs"
excerpt: "Understand TensorFlow 2.x's execution model — how eager mode works, when tf.function compiles to a graph, and what AutoGraph transforms automatically."
author: "Soham Sharma"
category: "Technology"
tags: ["TensorFlow", "Deep Learning", "Python", "Keras", "Machine Learning"]
status: "published"
featuredImage: ""
---

TensorFlow 1.x made you build a static computation graph first, then feed data through it in a session. It was fast but painful — you couldn't use Python debuggers, intermediate values were invisible, and errors surfaced at runtime with cryptic messages. TensorFlow 2.x flipped this: execution is eager by default, code runs immediately like normal Python. But there's a catch. When you need performance, you reach for `tf.function`, and suddenly you're dealing with tracing, retracing, and the AutoGraph compiler. Understanding when you're in eager mode vs. graph mode is the single most important mental model for working productively with TF2.

![TensorFlow 2.x architecture showing eager vs graph execution](https://www.tensorflow.org/images/tf_logo_social.png)

## Eager Execution: The Default Mode

In TF2, every operation executes immediately and returns a concrete value. No sessions, no placeholders, no `feed_dict`.

```python
import tensorflow as tf

# Eager execution — runs immediately
a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])
c = a + b

print(c)          # tf.Tensor([5. 7. 9.], shape=(3,), dtype=float32)
print(c.numpy())  # [5. 7. 9.]  — convert to NumPy array

# Operations work element-wise, shapes are known immediately
x = tf.random.normal([3, 4])
print(x.shape)    # (3, 4)
print(x.dtype)    # <dtype: 'float32'>
```

Eager mode means you can use Python control flow, print intermediate values, and use debuggers (pdb, ipdb) normally:

```python
def compute_softmax(logits):
    # Pure Python control flow works fine in eager mode
    if logits.shape[-1] == 1:
        return tf.sigmoid(logits)
    
    # Can inspect intermediate values with print()
    print(f"Input shape: {logits.shape}")
    exp_logits = tf.exp(logits - tf.reduce_max(logits, axis=-1, keepdims=True))
    print(f"Exp logits shape: {exp_logits.shape}")
    
    return exp_logits / tf.reduce_sum(exp_logits, axis=-1, keepdims=True)

logits = tf.constant([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
softmax = compute_softmax(logits)
print(softmax)
```

**The tradeoff**: eager execution is convenient but slower than compiled graphs. Python overhead adds up, and the GPU/CPU can't be efficiently pipelined. For interactive exploration and debugging, eager mode is ideal. For training loops and inference at scale, you need `tf.function`.

## tf.function: Compiling Python to Graphs

Decorating a Python function with `@tf.function` triggers **tracing** — TensorFlow executes the function once in a special mode, recording the operations into a `ConcreteFunction` (a compiled graph). Subsequent calls with the same input signature use the compiled graph, bypassing Python entirely.

```python
# Without tf.function — Python overhead on every call
def matrix_multiply(a, b):
    return tf.matmul(a, b)

# With tf.function — compiled to graph after first call
@tf.function
def matrix_multiply_fast(a, b):
    return tf.matmul(a, b)

import time

a = tf.random.normal([1000, 1000])
b = tf.random.normal([1000, 1000])

# Warm up
_ = matrix_multiply_fast(a, b)

# Benchmark
start = time.time()
for _ in range(100):
    matrix_multiply(a, b)
print(f"Eager: {time.time() - start:.3f}s")

start = time.time()
for _ in range(100):
    matrix_multiply_fast(a, b)
print(f"tf.function: {time.time() - start:.3f}s")
# tf.function is typically 2-10x faster for compute-bound operations
```

### Tracing: How tf.function Works

The first time you call a `@tf.function`, TensorFlow **traces** it: Python code runs, but TF operations don't produce concrete values — instead they produce **symbolic tensors** that represent the computation. The resulting graph is cached.

```python
@tf.function
def add_one(x):
    print(f"Tracing with x = {x}")  # Only runs during tracing
    return x + 1

a = tf.constant(1)
b = tf.constant(2)
c = tf.constant(3)

result1 = add_one(a)  # Prints: "Tracing with x = Tensor(...)"
result2 = add_one(b)  # No print — uses cached graph (same dtype/shape)
result3 = add_one(c)  # No print — still same signature
```

Tracing is triggered again (retracing) when the input signature changes in a way TensorFlow considers distinct:

```python
@tf.function
def process(x):
    print("Retracing!")
    return x * 2

# Each distinct Python value triggers a retrace (Python scalars are not symbolic)
process(1)      # Retrace: Python int 1
process(2)      # Retrace: Python int 2
process(3)      # Retrace: Python int 3

# TF tensors with same shape/dtype share a trace
process(tf.constant(1))  # Retrace: tf.int32 scalar
process(tf.constant(5))  # No retrace — same signature
```

**Performance trap**: passing Python scalars that vary causes excessive retracing. Use `tf.constant` or `input_signature` to control this:

```python
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    tf.TensorSpec(shape=[None, None], dtype=tf.float32)
])
def safe_matmul(a, b):
    return tf.matmul(a, b)

# Now shape changes within [None, None] don't retrace
# But incompatible shapes will raise an error at call time
```

## AutoGraph: Python → Graph Transformations

AutoGraph is the compiler that runs during tracing. It transforms Python control flow (`if`, `for`, `while`) into TensorFlow graph operations. Without AutoGraph, these constructs would be "burned in" to the graph with their Python values at trace time.

```python
@tf.function
def classify(x, threshold=0.5):
    # AutoGraph transforms this if/else into tf.cond
    if x > threshold:
        return tf.constant("positive")
    else:
        return tf.constant("negative")

# Both branches are compiled into the graph
print(classify(tf.constant(0.8)))  # positive
print(classify(tf.constant(0.2)))  # negative
```

Without AutoGraph, the `if` would evaluate at trace time using the symbolic tensor's value — which doesn't work. AutoGraph intercepts the `if` and replaces it with `tf.cond`, which evaluates both branches symbolically and selects at runtime.

```python
# AutoGraph transforms Python for loops over tf.Tensors
@tf.function
def sum_sequence(sequence):
    total = tf.constant(0.0)
    # AutoGraph converts this to tf.while_loop
    for element in sequence:
        total = total + element
    return total

seq = tf.constant([1.0, 2.0, 3.0, 4.0])
print(sum_sequence(seq))  # tf.Tensor(10.0, ...)
```

### AutoGraph Gotchas

AutoGraph is powerful but has sharp edges:

```python
# PROBLEM: Python list mutation inside tf.function
@tf.function
def bad_append(x):
    results = []           # Python list — not a TF tensor
    for i in tf.range(3):
        results.append(x * i)   # Traced 3 times but list is Python-level
    return results         # May not behave as expected

# SOLUTION: Use TensorArray for dynamic collections
@tf.function
def good_append(x):
    results = tf.TensorArray(dtype=tf.float32, size=3)
    for i in tf.range(3):
        results = results.write(i, x * tf.cast(i, tf.float32))
    return results.stack()

print(good_append(tf.constant(2.0)))  # [0. 2. 4.]
```

```python
# PROBLEM: Side effects in tf.function are not guaranteed
counter = tf.Variable(0)

@tf.function
def increment_bad():
    counter.assign_add(1)
    print(f"Counter is now {counter.numpy()}")  # Only runs during tracing!

increment_bad()  # Prints during trace
increment_bad()  # Doesn't print — graph is cached
print(counter)   # Value is 2, but print only ran once
```

The `print()` inside `@tf.function` runs only during tracing. Use `tf.print()` for output that executes at graph runtime:

```python
@tf.function
def increment_correct():
    counter.assign_add(1)
    tf.print("Counter is now:", counter)  # Runs every time graph executes

increment_correct()  # "Counter is now: 1"
increment_correct()  # "Counter is now: 2"
```

![TensorFlow function tracing and graph compilation process](https://www.tensorflow.org/guide/images/intro_to_graphs/two_types_of_functions.png)

## Variables: State in Graphs

`tf.Variable` is TensorFlow's stateful tensor — it persists across function calls and graph executions. Regular `tf.constant` tensors are immutable nodes in the graph.

```python
# tf.Variable — mutable, persists across calls
weights = tf.Variable(tf.random.normal([128, 64]), name="weights")
bias = tf.Variable(tf.zeros([64]), name="bias")

@tf.function
def linear_forward(x):
    return tf.matmul(x, weights) + bias

# Variables are updated in place
@tf.function
def update_weights(grad_w, grad_b, lr=0.01):
    weights.assign_sub(lr * grad_w)   # in-place subtract
    bias.assign_sub(lr * grad_b)

# In a training step
x = tf.random.normal([32, 128])
with tf.GradientTape() as tape:
    output = linear_forward(x)
    loss = tf.reduce_mean(tf.square(output))

grad_w, grad_b = tape.gradient(loss, [weights, bias])
update_weights(grad_w, grad_b)
```

## Concrete Functions and the tf.function Cache

Each unique trace produces a `ConcreteFunction`. You can inspect them:

```python
@tf.function
def multiply(a, b):
    return a * b

# Trace with different signatures
result1 = multiply(tf.constant(2.0), tf.constant(3.0))
result2 = multiply(tf.constant([1, 2]), tf.constant([3, 4]))

# Inspect concrete functions
print(multiply.pretty_printed_concrete_signatures())
# Shows each traced variant with its input signature
```

Each concrete function is a fully compiled XLA/TF graph. When TensorFlow decides which concrete function to use for a given call, it performs **dispatch** based on the argument signatures.

## When to Use tf.function (and When Not To)

Use `@tf.function` on:
- Training steps (`train_step`)
- Inference functions called thousands of times
- Any computation-heavy function that doesn't need Python-level debugging

Avoid `@tf.function` on:
- Functions you're actively debugging (eager is friendlier)
- Functions called only once or twice (tracing overhead exceeds benefit)
- Functions with complex Python-level logic that AutoGraph can't handle cleanly

```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(10)

    def call(self, x, training=False):
        x = self.dense(x, training=training)
        return self.output_layer(x)

model = MyModel()

@tf.function
def train_step(model, x, y, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions, from_logits=True)
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
x_batch = tf.random.normal([32, 128])
y_batch = tf.random.uniform([32], minval=0, maxval=10, dtype=tf.int32)

loss = train_step(model, x_batch, y_batch, optimizer)
print(f"Loss: {loss:.4f}")
```

## Conclusion

TensorFlow 2.x's execution model gives you the best of both worlds — eager execution for development velocity and `tf.function` for production performance. The key mental model: your code runs as Python first (tracing), then as a compiled graph (execution). Python `print()` runs during tracing; `tf.print()` runs during graph execution. Python control flow is frozen unless AutoGraph rewrites it. Variables persist; constants don't.

Once this distinction is clear, most TF2 debugging becomes mechanical: if a computation isn't happening when you expect, check whether you're in graph mode and whether AutoGraph has transformed your control flow correctly.
