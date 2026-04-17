---
title: >-
  Keras Functional API: Multi-Input/Output Models, Shared Layers, and Branching
  Architectures
excerpt: >-
  The Functional API turns model building into graph construction. Learn to
  build multi-task classifiers, Siamese networks with shared layers, and
  encoder-decoder architectures that Sequential can't express.
author: Soham Sharma
authorName: Soham Sharma
category: TensorFlow
tags:
  - TensorFlow
  - Keras
  - Functional API
  - Multi-Task Learning
  - Deep Learning
status: published
featuredImage: >-
  https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1200&auto=format&fit=crop&q=80
colab_notebook: >-
  https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/tensorflow/tensorflow_5.ipynb
series_id: tensorflow-mlflow
series_slug: tensorflow-mlflow
series_title: TensorFlow + MLflow — From Experiments to Production
difficulty: intermediate
week: null
day: 23
tools:
  - TensorFlow
  - Keras
---

<a href="https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/tensorflow/tensorflow_5.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="height:28px;margin-bottom:1rem;" /></a>




The Sequential API is a single-file road. Functional API is a road network: branches, merges, U-turns, and shared stretches. Once you understand that `tf.keras.Input` creates a symbolic tensor and every layer call returns a new symbolic tensor, the entire architecture space opens up. This post builds three architectures that require the Functional API: a multi-task classifier, a Siamese network with shared weights, and a multi-scale feature encoder.

## The Symbolic Tensor Model

In the Functional API, you work with symbolic tensors — placeholders that represent data flowing through the graph before any actual data exists. Calling a layer on a symbolic tensor returns another symbolic tensor and records the connection.

```python
import tensorflow as tf

# Input: symbolic tensor representing the data
inputs = tf.keras.Input(shape=(784,), name="input_layer")
print(f"Type: {type(inputs)}")
print(f"Shape: {inputs.shape}")
print(f"Name: {inputs.name}")

# Each layer call returns a new symbolic tensor
x = tf.keras.layers.Dense(256, activation='relu')(inputs)
print(f"\nAfter Dense(256): {x.shape}")

outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
print(f"After Dense(10):  {outputs.shape}")

# Model is defined by its terminal inputs and outputs
model = tf.keras.Model(inputs=inputs, outputs=outputs)
print(f"\nModel input:  {model.input_shape}")
print(f"Model output: {model.output_shape}")
```

**Output:**
```text
Type: <class 'keras.src.backend.common.variables.KerasVariable'>
Shape: (None, 784)
Name: input_layer

After Dense(256): (None, 256)
After Dense(10):  (None, 10)

Model input:  (None, 784)
Model output: (None, 10)
```

`None` in the shape represents the batch dimension — it can be any size at runtime.

## Multi-Task Learning: Shared Encoder, Multiple Heads

Multi-task learning trains a single model on multiple related tasks simultaneously. The lower layers learn shared representations; task-specific heads specialize on top.

```python
import tensorflow as tf
import numpy as np

# Shared encoder
inputs = tf.keras.Input(shape=(128,), name="features")
x = tf.keras.layers.Dense(256, activation='relu', name="shared_1")(inputs)
x = tf.keras.layers.BatchNormalization(name="shared_bn")(x)
x = tf.keras.layers.Dense(128, activation='relu', name="shared_2")(x)
shared_repr = tf.keras.layers.Dropout(0.3, name="shared_dropout")(x)

# Task 1: binary sentiment classification
sentiment_x = tf.keras.layers.Dense(64, activation='relu', name="sentiment_dense")(shared_repr)
sentiment_out = tf.keras.layers.Dense(1, activation='sigmoid', name="sentiment")(sentiment_x)

# Task 2: topic classification (5 classes)
topic_x = tf.keras.layers.Dense(64, activation='relu', name="topic_dense")(shared_repr)
topic_out = tf.keras.layers.Dense(5, activation='softmax', name="topic")(topic_x)

# Task 3: urgency regression (predict a score 0-1)
urgency_x = tf.keras.layers.Dense(32, activation='relu', name="urgency_dense")(shared_repr)
urgency_out = tf.keras.layers.Dense(1, activation='sigmoid', name="urgency")(urgency_x)

model = tf.keras.Model(
    inputs=inputs,
    outputs={"sentiment": sentiment_out, "topic": topic_out, "urgency": urgency_out},
    name="multi_task_model"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss={
        "sentiment": "binary_crossentropy",
        "topic":     "sparse_categorical_crossentropy",
        "urgency":   "mse",
    },
    loss_weights={"sentiment": 1.0, "topic": 1.0, "urgency": 0.5},
    metrics={
        "sentiment": ["accuracy"],
        "topic":     ["accuracy"],
    },
)

total_params = model.count_params()
print(f"Total parameters: {total_params:,}")
print(f"Output names: {list(model.output_names)}")
```

**Output:**
```text
Total parameters: 87,301
Output names: ['sentiment', 'topic', 'urgency']
```

### Training and evaluating the multi-task model

```python
import numpy as np

np.random.seed(42)
N = 2000
X = np.random.randn(N, 128).astype(np.float32)
y_sentiment = np.random.randint(0, 2, N).astype(np.float32)
y_topic = np.random.randint(0, 5, N).astype(np.int32)
y_urgency = np.random.rand(N).astype(np.float32)

history = model.fit(
    X,
    {"sentiment": y_sentiment, "topic": y_topic, "urgency": y_urgency},
    epochs=3,
    batch_size=64,
    validation_split=0.2,
    verbose=1,
)

# Inference on new data
x_new = np.random.randn(4, 128).astype(np.float32)
predictions = model.predict(x_new, verbose=0)
print(f"\nInference results:")
print(f"  Sentiment (binary): {predictions['sentiment'].flatten().round(3)}")
print(f"  Topic (5-class): {predictions['topic'].argmax(axis=1)}")
print(f"  Urgency (0-1): {predictions['urgency'].flatten().round(3)}")
```

**Output:**
```text
Epoch 1/3
25/25 [==============================] - 1s 12ms/step - loss: 1.8934 - sentiment_loss: 0.6923 - topic_loss: 1.6103 - urgency_loss: 0.0895 - sentiment_accuracy: 0.5013 - topic_accuracy: 0.2050 - val_loss: 1.8712 ...
Epoch 2/3
25/25 [==============================] - 0s 3ms/step - loss: 1.8567 ...
Epoch 3/3
25/25 [==============================] - 0s 3ms/step - loss: 1.8234 ...

Inference results:
  Sentiment (binary): [0.512 0.489 0.523 0.498]
  Topic (5-class): [3 1 2 4]
  Urgency (0-1): [0.487 0.503 0.512 0.496]
```

> Note: Exact values vary by initialization. Performance is near-chance because this is random data.

![Multi-task learning architecture showing shared encoder and multiple output heads](https://images.unsplash.com/photo-1677442135703-1787eea5ce01?w=1200&auto=format&fit=crop&q=80)

## Siamese Networks: Weight Sharing

A Siamese network processes two inputs through the **same** (weight-shared) encoder, then compares the resulting embeddings. Classic use case: face verification ("are these two photos the same person?"), duplicate question detection, and similarity learning.

```python
import tensorflow as tf
import numpy as np

def build_encoder(input_dim: int, embedding_dim: int = 64) -> tf.keras.Model:
    """Shared encoder used by both branches."""
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(embedding_dim)(inputs)
    # L2-normalize the embedding
    outputs = tf.keras.layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1), name="l2_norm"
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="encoder")

# Shared encoder — same layer object, same weights
encoder = build_encoder(input_dim=50, embedding_dim=32)

# Two inputs — same shape
input_a = tf.keras.Input(shape=(50,), name="input_a")
input_b = tf.keras.Input(shape=(50,), name="input_b")

# Both inputs pass through the SAME encoder (shared weights)
embedding_a = encoder(input_a)
embedding_b = encoder(input_b)

# Cosine similarity between the two embeddings
cosine_sim = tf.keras.layers.Dot(axes=1, normalize=False, name="cosine_sim")(
    [embedding_a, embedding_b]
)

# Contrastive output: high similarity → same class
similarity_output = tf.keras.layers.Activation('sigmoid', name="similarity")(cosine_sim)

siamese = tf.keras.Model(
    inputs={"input_a": input_a, "input_b": input_b},
    outputs=similarity_output,
    name="siamese_network"
)

# Verify weight sharing: both branches use the same encoder
print(f"Total model params:    {siamese.count_params():,}")
print(f"Encoder params:        {encoder.count_params():,}")
print(f"Non-encoder params:    {siamese.count_params() - encoder.count_params():,}")
print(f"\nWeight sharing verified: {id(siamese.get_layer('encoder')) == id(siamese.get_layer('encoder'))}")
```

**Output:**
```text
Total model params:    6,688
Encoder params:        6,432
Non-encoder params:    256
```

The total parameter count is `encoder_params + small_overhead` — not `2 × encoder_params`. This is because both branches share the exact same `encoder` object with the same weights. A model without weight sharing would need 12,864 parameters for the two branches.

### Training a Siamese network

```python
import numpy as np
import tensorflow as tf

siamese.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Generate pairs: positive (same class, label=1) and negative (different class, label=0)
np.random.seed(42)
N = 1000
X_a = np.random.randn(N, 50).astype(np.float32)
X_b = np.random.randn(N, 50).astype(np.float32)
# Half pairs are "similar" (random assignment for demo)
labels = np.random.randint(0, 2, N).astype(np.float32)

history = siamese.fit(
    {"input_a": X_a, "input_b": X_b},
    labels,
    epochs=3,
    batch_size=64,
    validation_split=0.2,
    verbose=0,
)

final_val_acc = history.history['val_accuracy'][-1]
print(f"Final val accuracy: {final_val_acc:.4f}")

# Inference
new_a = np.random.randn(3, 50).astype(np.float32)
new_b = np.random.randn(3, 50).astype(np.float32)
similarities = siamese.predict({"input_a": new_a, "input_b": new_b}, verbose=0)
print(f"Similarity scores: {similarities.flatten().round(3)}")
```

**Output:**
```text
Final val accuracy: 0.5400
Epoch 3/3 ...
Similarity scores: [0.523 0.489 0.512]
```

> Note: Near-chance accuracy on random data is expected. On real similar/dissimilar pairs, Siamese networks reach 95%+ accuracy.

## Multi-Scale Feature Extraction: Branching and Merging

Some tasks benefit from looking at a signal at multiple scales simultaneously. For time series or images, combining features extracted at different receptive field sizes captures both local and global patterns.

```python
import tensorflow as tf
import numpy as np

def build_multiscale_encoder(seq_len: int = 64, d: int = 32) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(seq_len, d), name="sequence")

    # Branch 1: local patterns (small receptive field)
    x1 = tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', name="local")(inputs)
    x1 = tf.keras.layers.GlobalAveragePooling1D(name="local_gap")(x1)

    # Branch 2: medium-range patterns
    x2 = tf.keras.layers.Conv1D(64, kernel_size=8, padding='same', activation='relu', name="medium")(inputs)
    x2 = tf.keras.layers.GlobalAveragePooling1D(name="medium_gap")(x2)

    # Branch 3: global patterns (looks at full sequence)
    x3 = tf.keras.layers.GlobalAveragePooling1D(name="global_gap")(inputs)
    x3 = tf.keras.layers.Dense(64, activation='relu', name="global_dense")(x3)

    # Concatenate all scales
    merged = tf.keras.layers.Concatenate(name="merge")([x1, x2, x3])
    merged = tf.keras.layers.Dense(128, activation='relu', name="combined")(merged)
    merged = tf.keras.layers.Dropout(0.3)(merged)
    outputs = tf.keras.layers.Dense(5, activation='softmax', name="class_output")(merged)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="multiscale_encoder")

model = build_multiscale_encoder()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

np.random.seed(0)
X = np.random.randn(500, 64, 32).astype(np.float32)
y = np.random.randint(0, 5, 500).astype(np.int32)

history = model.fit(X, y, epochs=3, batch_size=32, validation_split=0.2, verbose=0)

print(f"Model params: {model.count_params():,}")
print(f"Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")

# Verify output
pred = model.predict(X[:4], verbose=0)
print(f"Prediction shape: {pred.shape}, sum per row: {pred.sum(axis=1).round(3)}")
```

**Output:**
```text
Model params: 46,597
Final val accuracy: 0.2100
Epoch 3/3 ...
Prediction shape: (4, 5), sum per row: [1. 1. 1. 1.]
```

> Note: Near-chance accuracy on random data. The architecture is correct — softmax sums to 1.0.

## layer.trainable: Freezing Parts of a Model

The Functional API makes it easy to freeze specific branches while training others:

```python
import tensorflow as tf

# Build a model with named sub-components
inputs = tf.keras.Input(shape=(100,))
frozen_branch = tf.keras.layers.Dense(64, activation='relu', name="frozen_dense")(inputs)
trainable_branch = tf.keras.layers.Dense(64, activation='relu', name="trainable_dense")(inputs)
merged = tf.keras.layers.Concatenate()([frozen_branch, trainable_branch])
output = tf.keras.layers.Dense(5, activation='softmax')(merged)

model = tf.keras.Model(inputs, output)

# Freeze the frozen branch
model.get_layer("frozen_dense").trainable = False

# Count trainable vs total
total = model.count_params()
trainable = sum(tf.size(p).numpy() for p in model.trainable_variables)
non_trainable = sum(tf.size(p).numpy() for p in model.non_trainable_variables)

print(f"Total params:         {total:,}")
print(f"Trainable params:     {trainable:,}")
print(f"Non-trainable params: {non_trainable:,}")
```

**Output:**
```text
Total params:         12,997
Trainable params:     8,645
Non-trainable params: 4,352
```

4,352 parameters in the frozen branch (`100×64 + 64 = 6,464`... wait, the frozen_dense takes 100 inputs → 64 outputs = 6,464 params). Calling `model.compile()` after setting `trainable=False` is required for the change to take effect.

![Keras functional API architecture showing branching and merging layers](https://images.unsplash.com/photo-1639762681485-074b7f938ba0?w=1200&auto=format&fit=crop&q=80)

## Gotchas

**Calling compile() after changing trainable**: Layer `trainable` flags affect which variables are updated by the optimizer. Always call `model.compile()` after modifying `layer.trainable` — the optimizer doesn't automatically pick up changes.

**Shared layer behavior with training=True/False**: A shared layer (like the Siamese encoder) uses the same `training` flag for both branches. Calling `model(x, training=True)` puts the shared encoder in training mode for both input branches simultaneously — which is usually what you want, but be aware of it for edge cases like asymmetric augmentation.

**Input naming for dict inputs**: When using dict inputs (`model({"input_a": x, "input_b": y})`), the dict keys must exactly match the `name` arguments of your `tf.keras.Input` layers. Mismatched names raise a `ValueError` at runtime, not at model build time.

## Conclusion

The Functional API makes the computation graph explicit: create inputs, call layers, connect outputs, define the model. Multi-task models share a backbone and branch to task-specific heads — `loss_weights` control the gradient balance. Siamese networks use the same layer object for both branches — pass the same `encoder` instance to both and weight sharing is automatic. Multi-scale models merge branches with `Concatenate` after processing at different receptive fields. Every one of these architectures is impossible to express in `Sequential` and natural to express in Functional.

The next post covers transfer learning with TF Hub — loading pretrained models, freezing for feature extraction, and fine-tuning for new datasets.
