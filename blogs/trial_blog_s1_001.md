---
id: "TRIAL-S1-001"
title: "PyTorch from Scratch in Production: Tensors, Autograd, and a Deployable Mini Classifier"
series_id: "S1"
series_slug: "pytorch"
series_title: "PyTorch Mastery: From Tensors to Production"
difficulty: "beginner"
week: 1
day: 1
tags:
  - pytorch
  - deep-learning
  - autograd
  - production
  - fastapi
tools:
  - PyTorch
  - NumPy
  - FastAPI
  - Docker
word_count: 0
generated_at: "2026-04-13T00:00:00Z"
status: "draft"
published_to: []
---

> Series: PyTorch Mastery: From Tensors to Production | Trial Post

# PyTorch from Scratch in Production: Tensors, Autograd, and a Deployable Mini Classifier

## TL;DR

- Tensors are the core abstraction in PyTorch, and autograd is the engine that makes training possible.
- If you understand computation graphs, debugging exploding loss and broken gradients becomes much easier.
- You can go from notebook-level model code to a small production API in one clean pipeline.
- A strong technical blog should not stop at theory; it should include one complete build-and-deploy mini project.

---

## Introduction

Most PyTorch tutorials teach the happy path: define model, train model, print accuracy. That is useful, but incomplete.

Real engineering starts when things fail:

- Gradients become `None`
- Loss turns into `nan`
- Inference behaves differently from training
- A model works in a notebook but fails in an API service

This article gives you the practical base layer: tensors, autograd, and computation graph mechanics, then applies them to a deployable mini project so you can copy the exact workflow.

---

## Prerequisites

- Python 3.10+
- Basic linear algebra intuition
- Familiarity with Python classes/functions

Install dependencies:

```bash
# Core training stack
pip install torch==2.3.0 numpy==1.26.4 scikit-learn==1.5.0

# Serving stack
pip install fastapi==0.115.0 uvicorn==0.30.6 pydantic==2.8.2
```

---

## Core Concepts

### 1. Tensors are N-dimensional arrays with context

A PyTorch tensor is like a NumPy array plus:

- Device awareness (CPU/GPU)
- Dtype control
- Optional gradient tracking

```python
import torch

# 2x3 float tensor
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]], dtype=torch.float32)

print(x.shape)      # torch.Size([2, 3])
print(x.dtype)      # torch.float32
print(x.device)     # cpu (or cuda if moved)
```

### 2. `requires_grad=True` starts graph tracking

When a tensor requires grad, PyTorch records operations on it.

```python
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

x = torch.tensor(3.0)
y = w * x + b  # graph is created here
```

### 3. Backpropagation is reverse traversal of the graph

Calling `loss.backward()` computes gradients for every leaf tensor with `requires_grad=True`.

```python
loss = (y - 10.0) ** 2
loss.backward()

print(w.grad)  # d(loss)/d(w)
print(b.grad)  # d(loss)/d(b)
```

### 4. Common gradient pitfall

Gradients accumulate by default.

```python
optimizer.zero_grad()  # always clear previous gradients
loss.backward()        # compute current gradients
optimizer.step()       # update parameters
```

If you skip `zero_grad()`, your updates use stale history and training drifts.

---

## Implementation: Train a Real Mini Classifier

We will train a binary classifier using synthetic 2D data and export weights for serving.

```python
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate simple two-class dataset
# Class 0 centered near (-1, -1), class 1 near (1, 1)
n_samples = 2000
x0 = np.random.normal(loc=-1.0, scale=0.8, size=(n_samples // 2, 2))
x1 = np.random.normal(loc=1.0, scale=0.8, size=(n_samples // 2, 2))

X = np.vstack([x0, x1]).astype(np.float32)
y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)]).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train).unsqueeze(1)
X_test_t = torch.from_numpy(X_test)
y_test_t = torch.from_numpy(y_test).unsqueeze(1)

class MiniClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Small MLP for binary classification
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        # Raw logits output; sigmoid applied during evaluation
        return self.net(x)

model = MiniClassifier()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 80
for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()
    logits = model(X_train_t)
    loss = criterion(logits, y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d} | loss={loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
    test_probs = torch.sigmoid(test_logits)
    preds = (test_probs >= 0.5).float()

acc = accuracy_score(y_test_t.numpy(), preds.numpy())
print(f"Test accuracy: {acc:.4f}")

# Save model state for deployment
torch.save(model.state_dict(), "mini_classifier.pt")
print("Saved model -> mini_classifier.pt")
```

Expected result: accuracy around 0.95 to 0.99 on this synthetic setup.

---

## End-to-End Project Section (Build + Deploy)

This is the section you said you want across the series. Here is a complete pattern you can replicate for 100 projects.

### Project: Decision Boundary API

Goal: expose the trained classifier as an HTTP API that returns class probability.

`app.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

app = FastAPI(title="Mini Classifier API")

class MiniClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MiniClassifier()
model.load_state_dict(torch.load("mini_classifier.pt", map_location="cpu"))
model.eval()

class PredictRequest(BaseModel):
    x1: float
    x2: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    # Convert request values into model input tensor
    x = torch.tensor([[req.x1, req.x2]], dtype=torch.float32)

    # Inference should always be in no_grad mode for performance/safety
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()

    return {
        "probability_class_1": round(prob, 6),
        "predicted_class": int(prob >= 0.5)
    }
```

Run locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Test request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"x1": 1.2, "x2": 0.9}'
```

Dockerfile:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir torch==2.3.0 fastapi==0.115.0 uvicorn==0.30.6 pydantic==2.8.2

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t mini-classifier-api .
docker run -p 8000:8000 mini-classifier-api
```

Branch strategy for your 100 projects:

- Keep each project in its own branch: `project/001-mini-classifier`, `project/002-rag-chatbot`, etc.
- Blog post includes branch name + repo path for copy-and-run.
- Publish pipeline links article to its project branch and deployment URL.

---

## Production Notes

- Always separate training and inference code paths.
- Pin dependency versions in every project article.
- Add `/health` and input validation in every API example.
- Keep one `README.md` per project branch with setup, run, and deploy commands.
- Add simple CI test: model loads + one prediction route call.

---

## Common Mistakes

1. Training mode during inference

```python
# Wrong for inference
model.train()

# Correct
model.eval()
```

2. No `torch.no_grad()` in serving

```python
# Wrong: tracks graph unnecessarily
pred = model(x)

# Correct: no gradient graph, lower memory
with torch.no_grad():
    pred = model(x)
```

3. Mismatched preprocessing between train and serve

If you normalize features in training, the exact same transformation must run in API inference.

---

## Benchmarks (Typical Local Machine)

- Model size: tiny (< 50 KB state dict)
- Single prediction latency (CPU): ~1-3 ms
- Throughput with Uvicorn single worker: hundreds to low thousands req/s depending on hardware

For larger models, use batch inference + async queue + worker pool.

---

## Summary

You now have:

- A clear mental model of tensors and autograd
- A practical training loop with stable gradient flow
- A deployment-ready FastAPI inference service
- A repeatable article pattern that includes end-to-end project delivery

This is exactly the quality format you can scale across a 500-blog roadmap with 100 buildable projects.

---

## What’s Next

- Next post: training-loop diagnostics (vanishing gradients, exploding gradients, and debug hooks)
- Next project: experiment tracker integration with MLflow + model versioned serving
- Advanced extension: add CI pipeline to run unit tests and deploy branch preview automatically

---

## References

1. PyTorch Docs: https://pytorch.org/docs/stable/index.html
2. FastAPI Docs: https://fastapi.tiangolo.com/
3. Goodfellow et al., Deep Learning (for gradient/backprop fundamentals)
4. Practical MLOps patterns from open-source serving systems (TorchServe, BentoML)
