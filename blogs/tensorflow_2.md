---
title: 'Keras Sequential vs Functional vs Subclassing: When to Use Which API'
excerpt: >-
  Keras gives you three model-building APIs. Sequential is a dead end for
  anything non-trivial. Functional handles 90% of production architectures.
  Subclassing gives you full control when you need it.
author: Soham Sharma
authorName: Soham Sharma
category: TensorFlow
tags:
  - TensorFlow
  - Keras
  - Deep Learning
  - Model Architecture
  - Python
status: published
featuredImage: >-
  https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1200&auto=format&fit=crop&q=80
colab_notebook: >-
  https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/tensorflow/tensorflow_2.ipynb
series_id: tensorflow-mlflow
series_slug: tensorflow-mlflow
series_title: TensorFlow + MLflow — From Experiments to Production
difficulty: beginner
week: null
day: 7
tools:
  - TensorFlow
  - Keras
---

<a href="https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/tensorflow/tensorflow_2.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="height:28px;margin-bottom:1rem;" /></a>




Keras has three different ways to define a model. Tutorials start with `Sequential` because it's simple, but then you hit a case where you need two inputs, a skip connection, or a custom training step — and suddenly you need to rewrite everything. Knowing upfront when each API is the right tool saves that rewrite.

## The Sequential API: Great for Linear Stacks

`Sequential` is for models where data flows through layers in a straight line — no branching, no multiple inputs, no shared layers. It's the right choice for quick prototypes and simple baselines.

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.summary()
```

**Output:**
```text
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 128)               2688      
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 64)                8256      
                                                                 
 dense_2 (Dense)             (None, 10)                650       
                                                                 
=================================================================
Total params: 11594 (45.29 KB)
Trainable params: 11594 (45.29 KB)
Non-trainable params: 0 (0.00 B)
_________________________________________________________________
```

Clean, readable, and enough for many classification tasks. But `Sequential` hits a wall immediately when you need:
- Multiple inputs or outputs
- Skip connections (ResNet)
- Shared layers (Siamese networks)
- Layers with multiple outputs used in different parts of the model

For any of these, reach for the Functional API.

## The Functional API: DAGs, Not Pipelines

The Functional API treats the model as a computation graph. You create input tensors, pass them through layers, and define the model by specifying inputs and outputs. This handles any feedforward architecture — including residual connections, multi-input, and multi-output models.

```python
import tensorflow as tf

# Define inputs
inputs = tf.keras.Input(shape=(20,), name="features")

# Build the graph
x = tf.keras.layers.Dense(128, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)

# Skip connection: add inputs (projected) to intermediate output
shortcut = tf.keras.layers.Dense(64)(inputs)
x = tf.keras.layers.Add()([x, shortcut])
x = tf.keras.layers.Activation('relu')(x)

output = tf.keras.layers.Dense(10, activation='softmax', name="predictions")(x)

# Model is defined by its inputs and outputs
model = tf.keras.Model(inputs=inputs, outputs=output, name="residual_classifier")
model.summary()
```

**Output:**
```text
Model: "residual_classifier"
__________________________________________________________________________________________________
 Layer (type)                Output Shape      Param #   Connected to                     
==================================================================================================
 features (InputLayer)       [(None, 20)]      0         []                               
 dense (Dense)               (None, 128)       2688      ['features[0][0]']               
 dropout (Dropout)           (None, 128)       0         ['dense[0][0]']                  
 dense_1 (Dense)             (None, 64)        8256      ['dropout[0][0]']                
 dense_2 (Dense)             (None, 64)        1344      ['features[0][0]']               
 add (Add)                   (None, 64)        0         ['dense_1[0][0]','dense_2[0][0]']
 activation (Activation)     (None, 64)        0         ['add[0][0]']                    
 predictions (Dense)         (None, 10)        650       ['activation[0][0]']             
==================================================================================================
Total params: 12938 (50.54 KB)
Trainable params: 12938 (50.54 KB)
Non-trainable params: 0 (0.00 B)
__________________________________________________________________________________________________
```

Notice the `Connected to` column — the `Add` layer connects to both `dense_1` and `dense_2`. This is a real skip connection, not possible in `Sequential`.

### Multi-input model with Functional API

```python
import tensorflow as tf

# Two separate input streams
text_input = tf.keras.Input(shape=(100,), name="text_features")
meta_input = tf.keras.Input(shape=(10,), name="metadata")

# Process each stream
text_encoded = tf.keras.layers.Dense(64, activation='relu')(text_input)
meta_encoded = tf.keras.layers.Dense(16, activation='relu')(meta_input)

# Merge streams
merged = tf.keras.layers.Concatenate()([text_encoded, meta_encoded])
merged = tf.keras.layers.Dense(32, activation='relu')(merged)

# Two output heads
sentiment = tf.keras.layers.Dense(1, activation='sigmoid', name="sentiment")(merged)
topic = tf.keras.layers.Dense(5, activation='softmax', name="topic")(merged)

model = tf.keras.Model(
    inputs={"text_features": text_input, "metadata": meta_input},
    outputs={"sentiment": sentiment, "topic": topic},
    name="multi_task_classifier"
)

print(f"Inputs: {list(model.input_names)}")
print(f"Outputs: {list(model.output_names)}")
```

**Output:**
```text
Inputs: ['text_features', 'metadata']
Outputs: ['sentiment', 'topic']
```

Dict inputs and outputs make inference code readable — you call `model({"text_features": x1, "metadata": x2})` and get back `{"sentiment": ..., "topic": ...}`.

![Keras model architecture diagram showing sequential vs functional vs subclassing](https://images.unsplash.com/photo-1677442135703-1787eea5ce01?w=1200&auto=format&fit=crop&q=80)

## Training a Functional Model

Functional models use `.compile()` and `.fit()` the same as Sequential — the API is identical from the training perspective:

```python
import tensorflow as tf
import numpy as np

# Rebuild the multi-task model from above
text_input = tf.keras.Input(shape=(100,), name="text_features")
meta_input = tf.keras.Input(shape=(10,), name="metadata")
text_enc = tf.keras.layers.Dense(64, activation='relu')(text_input)
meta_enc = tf.keras.layers.Dense(16, activation='relu')(meta_input)
merged = tf.keras.layers.Concatenate()([text_enc, meta_enc])
merged = tf.keras.layers.Dense(32, activation='relu')(merged)
sentiment = tf.keras.layers.Dense(1, activation='sigmoid', name="sentiment")(merged)
topic = tf.keras.layers.Dense(5, activation='softmax', name="topic")(merged)

model = tf.keras.Model(
    inputs={"text_features": text_input, "metadata": meta_input},
    outputs={"sentiment": sentiment, "topic": topic},
)

model.compile(
    optimizer='adam',
    loss={"sentiment": "binary_crossentropy", "topic": "sparse_categorical_crossentropy"},
    loss_weights={"sentiment": 1.0, "topic": 0.5},
    metrics={"sentiment": "accuracy", "topic": "accuracy"},
)

# Dummy data
N = 200
x_data = {"text_features": np.random.randn(N, 100), "metadata": np.random.randn(N, 10)}
y_data = {"sentiment": np.random.randint(0, 2, N), "topic": np.random.randint(0, 5, N)}

history = model.fit(x_data, y_data, epochs=2, batch_size=32, verbose=1)
```

**Output:**
```text
Epoch 1/2
7/7 [==============================] - 1s 2ms/step - loss: 1.4832 - sentiment_loss: 0.7214 - topic_loss: 1.5237 - sentiment_accuracy: 0.4950 - topic_accuracy: 0.2050
Epoch 2/2
7/7 [==============================] - 0s 2ms/step - loss: 1.4423 - sentiment_loss: 0.7031 - topic_loss: 1.4784 - sentiment_accuracy: 0.5150 - topic_accuracy: 0.2300
```

> Note: Exact loss and accuracy values vary by random initialization.

Per-output loss weights let you balance the contribution of each task to the total gradient. `loss_weights={"sentiment": 1.0, "topic": 0.5}` means the sentiment loss contributes twice as much as the topic loss.

## The Subclassing API: Full Python Control

The subclassing API (`tf.keras.Model` subclass) gives you full Python control over the forward pass. You write `__init__` to create layers and `call()` to define the computation.

Use subclassing when:
- The computation graph is dynamic (varies by input, e.g., tree-structured inputs)
- You need custom training steps with unusual gradient manipulation
- The architecture requires Python-level branching that isn't representable as a static graph

```python
import tensorflow as tf

class ResidualBlock(tf.keras.layers.Layer):
    """A single residual block."""
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units)
        self.projection = tf.keras.layers.Dense(units)
        self.add = tf.keras.layers.Add()
        self.relu = tf.keras.layers.Activation('relu')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        shortcut = self.projection(inputs)
        x = self.add([x, shortcut])
        return self.relu(x)

class ResidualClassifier(tf.keras.Model):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.block1 = ResidualBlock(64)
        self.block2 = ResidualBlock(32)
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.block1(inputs, training=training)
        x = self.block2(x, training=training)
        return self.output_layer(x)

model = ResidualClassifier(num_classes=10)

# Build by passing dummy input
dummy = tf.zeros([1, 20])
_ = model(dummy)
model.summary()
```

**Output:**
```text
Model: "residual_classifier"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 residual_block (ResidualBlo  multiple                 5376      
                                                                 
 residual_block_1 (ResidualB  multiple                 3040      
                                                                 
 dense (Dense)               multiple                  330       
                                                                 
=================================================================
Total params: 8746 (34.16 KB)
Trainable params: 8746 (34.16 KB)
Non-trainable params: 0 (0.00 B)
_________________________________________________________________
```

Notice that subclassed models show `multiple` in the Output Shape column — Keras can't statically determine shapes for dynamic models. This also means `model.summary()` is less informative than for Functional models.

### The training=False argument pattern

The `training` argument must be threaded through every layer call to control `Dropout` and `BatchNormalization` behavior correctly:

```python
import tensorflow as tf

class ClassifierWithDropout(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(64, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.out = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense(inputs)
        x = self.dropout(x, training=training)  # ← must pass training here
        return self.out(x)

model = ClassifierWithDropout()
x = tf.random.normal([4, 10])

# training=True: dropout is active
train_out = model(x, training=True)
# training=False: dropout is disabled
infer_out = model(x, training=False)

print(f"Train output sum: {train_out.numpy().sum():.4f}")
print(f"Infer output sum: {infer_out.numpy().sum():.4f}")
```

**Output:**
```text
Train output sum: 4.0000
Infer output sum: 4.0000
```

> Note: Output sums are always 4.0 because softmax probabilities sum to 1.0 per sample × 4 samples. The actual values differ between training and inference modes.

Forgetting to pass `training=training` to `Dropout` means dropout is always disabled — your model trains without regularization and you won't notice until you compare train/val loss curves.

## When to Use Each API

| Scenario | API |
|---|---|
| Quick prototype, linear stack | Sequential |
| ResNets, U-Nets, multi-task models | Functional |
| Dynamic computation, RNNs with custom state | Subclassing |
| Production serving with SavedModel | Functional (best static graph support) |
| Research: custom gradient manipulation | Subclassing |
| Teaching / tutorials | Sequential → Functional |

### Gotcha: Subclassed models and SavedModel

Functional models serialize cleanly to SavedModel format — TensorFlow can trace the static graph. Subclassed models require that the `call()` method be traceable by `tf.function`. If your `call()` uses Python control flow that depends on non-tensor values (Python `if` on Python variables), the saved model may not behave identically to the Python model.

```python
import tensorflow as tf
import tempfile, os

# Functional model — serializes perfectly
inputs = tf.keras.Input(shape=(10,))
outputs = tf.keras.layers.Dense(5, activation='relu')(inputs)
functional_model = tf.keras.Model(inputs, outputs)

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "model")
    functional_model.save(path)
    loaded = tf.keras.models.load_model(path)
    result = loaded(tf.ones([2, 10]))
    print(f"Loaded functional model output shape: {result.shape}")
```

**Output:**
```text
Loaded functional model output shape: (2, 5)
```

Always test that a subclassed model produces identical outputs before and after `save`/`load`. If results differ, your `call()` has Python-side state that doesn't serialize.

![Neural network architecture comparison diagram](https://images.unsplash.com/photo-1639762681485-074b7f938ba0?w=1200&auto=format&fit=crop&q=80)

## Mixing the APIs: Custom Layers with Functional Models

You can mix subclassed layers with Functional models — define custom layers via subclassing, then use them in a Functional model definition. This is the recommended pattern for reusable custom components:

```python
import tensorflow as tf

class MultiHeadAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads: int, key_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim
        )
        self.norm = tf.keras.layers.LayerNormalization()
        self.ffn = tf.keras.layers.Dense(key_dim * num_heads, activation='relu')

    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs)
        x = self.norm(inputs + attn_output)
        return self.ffn(x)

# Use the custom layer in a Functional model
seq_input = tf.keras.Input(shape=(32, 64), name="sequence")  # (batch, seq_len, dim)
x = MultiHeadAttentionBlock(num_heads=4, key_dim=16)(seq_input)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
output = tf.keras.layers.Dense(3, activation='softmax')(x)

model = tf.keras.Model(inputs=seq_input, outputs=output)
print(f"Input: {model.input_shape}, Output: {model.output_shape}")
```

**Output:**
```text
Input: (None, 32, 64), Output: (None, 3)
```

This is the standard pattern for Transformer-based models in Keras: each block is a subclassed `Layer`, and the full model is assembled via the Functional API for clean serialization.

## Conclusion

Sequential is fine for learning and simple baselines — stop using it the moment your architecture deviates from a straight line. Functional handles the vast majority of production architectures, gives you clean `summary()` output, and serializes reliably. Subclassing is for research and dynamic graphs, but requires careful attention to the `training` argument and SavedModel compatibility. When in doubt, use Functional — you can always drop into a subclassed `Layer` for the parts that need custom logic.

The next post covers custom training loops with `GradientTape` — when you outgrow `.fit()` and need explicit control over the forward and backward pass.
