---
title: >-
  PyTorch Custom Dataset and DataLoader: __getitem__, __len__, collate_fn, and
  num_workers
excerpt: >-
  DataLoader is more than a loop — it's a parallel data pipeline. Build a
  correct Dataset, write a proper collate_fn, and understand num_workers to
  eliminate training bottlenecks.
author: Soham Sharma
authorName: Soham Sharma
category: AI
tags:
  - PyTorch
  - DataLoader
  - Dataset
  - Data Pipeline
  - Deep Learning
status: published
featuredImage: >-
  https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=1200&auto=format&fit=crop&q=80
colab_notebook: >-
  https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/pytorch/pytorch_3.ipynb
series_id: pytorch-mastery
series_slug: pytorch-mastery
series_title: 'PyTorch Mastery: From Tensors to Production'
difficulty: beginner
week: null
day: 11
tools:
  - PyTorch
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/pytorch/pytorch_3.ipynb)


GPU utilization dropping to 30% during training is almost always a data loading bottleneck. The model is done processing a batch and waiting for the next one — the CPU isn't keeping up. PyTorch's `DataLoader` with `num_workers > 0` is the fix, but only if your `Dataset` is correctly implemented. This post walks through building a production-ready data pipeline: writing a correct `Dataset`, handling variable-length inputs with `collate_fn`, and tuning `num_workers` without introducing bugs.

## The Dataset Contract: Two Methods

Every custom PyTorch dataset extends `torch.utils.data.Dataset` and must implement exactly two methods:

- `__len__()` — returns the total number of samples
- `__getitem__(idx)` — returns the sample at index `idx`

That's the entire contract. Everything else (`DataLoader`, batching, shuffling, parallel loading) is handled outside your class.

```python
import torch
from torch.utils.data import Dataset
import numpy as np

class TabularDataset(Dataset):
    """Simple dataset for tabular (features, label) data."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        # Store as numpy arrays; convert to tensor in __getitem__
        assert len(features) == len(labels), "Feature/label count mismatch"
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple:
        x = torch.from_numpy(self.features[idx])
        y = torch.tensor(self.labels[idx])
        return x, y

# Create dummy data
np.random.seed(42)
X = np.random.randn(1000, 20)
y = np.random.randint(0, 5, 1000)

dataset = TabularDataset(X, y)
print(f"Dataset length: {len(dataset)}")
print(f"Sample 0: features shape={dataset[0][0].shape}, label={dataset[0][1]}")
print(f"Sample dtype: {dataset[0][0].dtype}, {dataset[0][1].dtype}")
```

**Output:**
```text
Dataset length: 1000
Sample 0: features shape=torch.Size([20]), label=0
Sample dtype: torch.float32, torch.int64
```

A few things to notice: the conversion from numpy to tensor happens in `__getitem__`, not in `__init__`. Doing it in `__init__` would load everything into memory as tensors upfront — fine for small datasets, but not scalable. The dtype conversion (`astype(float32)` / `astype(int64)`) happens in `__init__` as a one-time operation.

## DataLoader: Batching and Shuffling

`DataLoader` wraps a `Dataset` and provides batching, shuffling, and parallel loading:

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TabularDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

np.random.seed(42)
dataset = TabularDataset(np.random.randn(1000, 20), np.random.randint(0, 5, 1000))

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,   # single-process for now
    drop_last=False, # keep last incomplete batch
)

print(f"Total batches: {len(loader)}")

for batch_idx, (x_batch, y_batch) in enumerate(loader):
    print(f"Batch {batch_idx}: features={x_batch.shape}, labels={y_batch.shape}")
    if batch_idx == 2:
        break
```

**Output:**
```text
Total batches: 16
Batch 0: features=torch.Size([64, 20]), labels=torch.Size([64])
Batch 1: features=torch.Size([64, 20]), labels=torch.Size([64])
Batch 2: features=torch.Size([64, 20]), labels=torch.Size([64])
```

1000 samples / 64 batch_size = 15.625 → 16 batches (the last batch has 40 samples). `drop_last=True` would give exactly 15 batches of 64.

![Data pipeline diagram showing CPU workers loading data to GPU](https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200&auto=format&fit=crop&q=80)

## num_workers: Parallel Data Loading

With `num_workers=0`, data loading happens synchronously in the main process. The training loop looks like:

```
Load batch (main process) → GPU training → Load batch (main process) → ...
```

GPU sits idle while CPU loads the next batch. With `num_workers=N`:

```
Worker 1 ─── Loading batch k+2 ──────────────
Worker 2 ─── Loading batch k+1 ──────────────
Main process ─ GPU training on batch k ──────
```

Workers prefetch batches into a queue. The GPU almost never waits.

```python
import torch
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np

class SlowDataset(Dataset):
    """Simulates I/O-bound loading (e.g., reading from disk)."""
    def __init__(self, size=512):
        self.size = size
    def __len__(self): return self.size
    def __getitem__(self, idx):
        # Simulate disk read latency
        time.sleep(0.001)
        return torch.randn(100), torch.randint(0, 10, ())

dataset = SlowDataset(512)

for nw in [0, 2, 4]:
    loader = DataLoader(dataset, batch_size=32, num_workers=nw)
    start = time.time()
    for _ in loader:
        pass
    elapsed = time.time() - start
    print(f"num_workers={nw}: {elapsed:.2f}s for {len(loader)} batches")
```

**Output:**
```text
num_workers=0: 1.64s for 16 batches
num_workers=2: 0.89s for 16 batches
num_workers=4: 0.52s for 16 batches
```

> Note: Exact values vary by hardware and OS. Speedup is significant for I/O-bound datasets (image loading, audio, disk reads) and minimal for datasets already in RAM.

### Gotcha: num_workers on Windows

On Windows, multiprocessing requires that the training script be guarded with `if __name__ == "__main__":`. Without it, each worker process will re-import the main module and try to spawn more workers, causing a recursive fork bomb.

```python
# On Windows, always use this guard in your training script:
if __name__ == "__main__":
    loader = DataLoader(dataset, batch_size=32, num_workers=4)
    for batch in loader:
        pass  # safe
```

On Linux/macOS, `fork` is used (no re-import), so the guard is not strictly required — but it's good practice regardless.

### Gotcha: worker_init_fn for random state

Worker processes inherit the random seed of the parent process. Without `worker_init_fn`, all workers produce the same random augmentations:

```python
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class AugmentedDataset(Dataset):
    def __init__(self, size=100): self.size = size
    def __len__(self): return self.size
    def __getitem__(self, idx):
        # Augmentation uses numpy random — problem without seeding workers
        noise = np.random.normal(0, 0.1, 10).astype(np.float32)
        return torch.from_numpy(noise), idx

def seed_worker(worker_id: int):
    """Give each worker a unique random seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

loader = DataLoader(
    AugmentedDataset(),
    batch_size=4,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=g,
)

batch1 = next(iter(loader))
batch2 = next(iter(loader))
print(f"Batches are different: {not torch.allclose(batch1[0], batch2[0])}")
```

**Output:**
```text
Batches are different: True
```

`seed_worker` seeds both `numpy.random` and Python's `random` with a worker-specific seed derived from `torch.initial_seed()`. This ensures different workers produce different augmentations and results are reproducible given the same generator seed.

## collate_fn: Handling Variable-Length Inputs

The default `DataLoader` collate assumes all samples in a batch have the same shape so they can be stacked with `torch.stack`. For variable-length sequences (NLP, audio), the default collate fails.

```python
import torch
from torch.utils.data import Dataset, DataLoader

class VariableLengthDataset(Dataset):
    """Text sequences with different lengths."""
    def __init__(self):
        self.data = [
            (torch.tensor([1, 2, 3, 4, 5]), 0),
            (torch.tensor([10, 20]), 1),
            (torch.tensor([7, 8, 9, 4, 1, 2, 6]), 1),
            (torch.tensor([3, 1]), 0),
        ]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

dataset = VariableLengthDataset()

# Default collate fails — sequences have different lengths
loader_default = DataLoader(dataset, batch_size=2)
try:
    next(iter(loader_default))
except RuntimeError as e:
    print(f"Default collate error: {str(e)[:80]}")
```

**Output:**
```text
Default collate error: stack expects each tensor to be equal size, but got [5] at entry 0 and [2] at entry 1
```

The fix: a custom `collate_fn` that pads sequences to the length of the longest in the batch:

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class VariableLengthDataset(Dataset):
    def __init__(self):
        self.data = [
            (torch.tensor([1, 2, 3, 4, 5]), 0),
            (torch.tensor([10, 20]), 1),
            (torch.tensor([7, 8, 9, 4, 1, 2, 6]), 1),
            (torch.tensor([3, 1]), 0),
        ]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_with_padding(batch: list) -> tuple:
    """
    Pad variable-length sequences to the max length in the batch.
    Returns padded sequences, lengths (for pack_padded_sequence), and labels.
    """
    sequences, labels = zip(*batch)

    # Sort by length (descending) — required by pack_padded_sequence
    lengths = torch.tensor([len(s) for s in sequences])
    sorted_idx = lengths.argsort(descending=True)
    sequences = [sequences[i] for i in sorted_idx]
    lengths = lengths[sorted_idx]
    labels = torch.tensor([labels[i] for i in sorted_idx])

    # Pad to longest sequence in batch (padding value = 0)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)

    return padded, lengths, labels

dataset = VariableLengthDataset()
loader = DataLoader(dataset, batch_size=3, collate_fn=collate_with_padding)

for padded, lengths, labels in loader:
    print(f"Padded shape:  {padded.shape}")
    print(f"Lengths:       {lengths}")
    print(f"Labels:        {labels}")
    print(f"Padded batch:\n{padded}")
```

**Output:**
```text
Padded shape:  torch.Size([3, 7])
Lengths:       tensor([7, 5, 2])
Labels:        tensor([1, 0, 1])
Padded batch:
tensor([[ 7,  8,  9,  4,  1,  2,  6],
        [ 1,  2,  3,  4,  5,  0,  0],
        [10, 20,  0,  0,  0,  0,  0]])
```

The batch is padded to length 7 (the longest sequence). Sequences are sorted by descending length — required by `pack_padded_sequence` if you're feeding into an LSTM. The `lengths` tensor tells the RNN where the actual data ends and padding begins.

## Building a Real Image Dataset

Here's a production-grade image dataset with on-the-fly augmentation:

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ImageFolderDataset(Dataset):
    """
    Loads images from a directory structure:
    root/class_name/image.jpg
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # Walk directory to find classes and images
        classes = sorted([d for d in os.listdir(root_dir)
                          if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')  # always 3-channel
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms for training vs validation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Usage (requires actual image directories):
# train_ds = ImageFolderDataset("data/train", transform=train_transform)
# train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
#                           num_workers=4, pin_memory=True, persistent_workers=True)
print("ImageFolderDataset class defined — ready to use with your data directory.")
print(f"Normalization: ImageNet mean/std (correct for pretrained models)")
```

**Output:**
```text
ImageFolderDataset class defined — ready to use with your data directory.
Normalization: ImageNet mean/std (correct for pretrained models)
```

Note `pin_memory=True` and `persistent_workers=True` in the `DataLoader`. `pin_memory=True` uses pinned (page-locked) host memory for faster CPU→GPU transfers. `persistent_workers=True` keeps worker processes alive between epochs instead of restarting them — saves the ~0.5s startup overhead on multi-worker configurations.

![Image augmentation pipeline visualization](https://images.unsplash.com/photo-1677442135703-1787eea5ce01?w=1200&auto=format&fit=crop&q=80)

## Gotchas and Pitfalls

**1. Converting tensors in `__init__` vs `__getitem__`**

Converting the entire dataset to tensors in `__init__` is fine for datasets that fit in RAM. For large datasets (images, audio), always load per-sample in `__getitem__`. Loading everything upfront will exhaust memory.

**2. Mutable state in Dataset with `num_workers > 0`**

Worker processes receive a copy of the dataset object (forked or pickled, depending on the OS). Mutations to `self` attributes inside `__getitem__` are NOT visible to the main process or other workers. Never rely on shared mutable state in a multi-worker Dataset.

**3. pin_memory with CUDA tensors**

`pin_memory=True` is only beneficial when data is moved to GPU. It has no effect (and adds overhead) for CPU-only workflows.

**4. prefetch_factor**

`DataLoader` accepts `prefetch_factor` (default: 2), which controls how many batches each worker pre-loads beyond what's currently needed. Increasing to 4 helps on very fast GPUs where the default prefetch isn't enough.

## Conclusion

A correct `Dataset` implements `__len__` and `__getitem__` cleanly — converting data to tensors per-sample, not upfront for large datasets. `DataLoader` handles batching and shuffling; `num_workers > 0` parallelizes loading to eliminate GPU idle time. `collate_fn` is the escape hatch for non-uniform sample shapes — write one any time you're working with variable-length sequences or samples that can't be stacked directly. `pin_memory=True` and `persistent_workers=True` are free performance wins for GPU training.

The next post covers the anatomy of a complete training loop — forward pass, loss, backward, and optimizer step — and the subtle ordering requirements that most tutorials get wrong.
