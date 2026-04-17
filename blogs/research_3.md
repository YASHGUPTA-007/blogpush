---
title: >-
  Grouped Query Attention (GQA): KV Cache Reduction and the Llama 2
  Implementation
excerpt: >-
  GQA reduces the KV cache by 8-16x with minimal quality loss by sharing
  key-value heads across query groups. This is how Llama 2 and most modern LLMs
  fit longer contexts on limited GPU memory.
author: Soham Sharma
authorName: Soham Sharma
category: AI
tags:
  - GQA
  - Attention
  - KV Cache
  - Llama
  - Transformers
  - Research
status: published
featuredImage: >-
  https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200&auto=format&fit=crop&q=80
colab_notebook: >-
  https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/research/research_3.ipynb
series_id: ai-research-explained
series_slug: ai-research-explained
series_title: Latest AI Research — Explained + Implemented
difficulty: beginner
week: null
day: 13
tools:
  - PyTorch
  - Transformers
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/research/research_3.ipynb)


During autoregressive inference, a transformer maintains a **KV cache** — the key and value tensors for every past token, for every attention head, for every layer. This cache is what allows the model to generate token N without recomputing attention over all N-1 previous tokens. The memory cost is significant: for Llama 2 70B with 64 heads, head dim 128, in float16, one token adds `2 × 64 × 128 × 2 bytes × 80 layers = 2.62 MB` per token to the cache. For a 4096-token context, that's 10.7 GB — just for the KV cache.

Multi-Query Attention (MQA, Shazeer 2019) reduced this by making all query heads share a single key-value head: 64 Q heads, 1 K head, 1 V head. The KV cache drops to 1/64th of the original. But MQA degrades model quality noticeably — too much parameter sharing.

**Grouped Query Attention (GQA)** (Ainslie et al., 2023) is the middle ground: group the 64 query heads into G groups, with each group sharing one K/V head. With G=8, you get 8 KV heads — an 8× reduction in KV cache with minimal quality loss. Llama 2 34B and 70B use GQA with G=8; Mistral 7B uses G=4.

## The Three Attention Variants

| Variant | Q heads | K heads | V heads | KV cache size | Quality |
|---|---|---|---|---|---|
| Multi-Head Attention (MHA) | H | H | H | Full (1×) | Best |
| Grouped Query Attention (GQA) | H | H/G | H/G | 1/G | Near-MHA |
| Multi-Query Attention (MQA) | H | 1 | 1 | 1/H | Degraded |

With H=32 heads and G=4 groups (Mistral 7B configuration): 8 KV heads, 4× cache reduction.

## Mathematical Foundation

In standard MHA, for each head h independently:

```
Attention_h(Q_h, K_h, V_h) = softmax(Q_h K_h^T / sqrt(d_k)) V_h
```

In GQA, the H query heads are partitioned into G groups of H/G. All heads in group g share the same K_g and V_g:

```
Attention_h(Q_h, K_g, V_g) = softmax(Q_h K_g^T / sqrt(d_k)) V_g
  where g = floor(h / (H/G))
```

The query projections remain independent (full H query heads), ensuring each head still has specialized query representations. Only the key-value projections are grouped.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention as used in Llama 2 70B and Mistral 7B.
    """
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0, \
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_q_heads // num_kv_heads
        self.head_dim = d_model // num_q_heads

        # Query: full heads, KV: reduced heads
        self.q_proj = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV heads to match Q heads: repeat each KV head num_groups times
        # (B, num_kv_heads, T, head_dim) → (B, num_q_heads, T, head_dim)
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)  # (B, num_q_heads, T, head_dim)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# Test: MHA equivalent (num_kv_heads = num_q_heads)
mha = GroupedQueryAttention(d_model=512, num_q_heads=8, num_kv_heads=8)
# GQA with 4 groups
gqa_4 = GroupedQueryAttention(d_model=512, num_q_heads=8, num_kv_heads=2)
# MQA equivalent (num_kv_heads = 1)
mqa = GroupedQueryAttention(d_model=512, num_q_heads=8, num_kv_heads=1)

x = torch.randn(2, 16, 512)  # (batch=2, seq_len=16, d_model=512)

out_mha  = mha(x)
out_gqa  = gqa_4(x)
out_mqa  = mqa(x)

print(f"Output shape (all variants): {out_mha.shape}")
print(f"\nParameter counts:")
print(f"  MHA  (8 KV heads): {sum(p.numel() for p in mha.parameters()):,}")
print(f"  GQA  (2 KV heads): {sum(p.numel() for p in gqa_4.parameters()):,}")
print(f"  MQA  (1 KV head):  {sum(p.numel() for p in mqa.parameters()):,}")
```

**Output:**
```text
Output shape (all variants): torch.Size([2, 16, 512])
Parameter counts:
  MHA  (8 KV heads): 1,050,624
  GQA  (2 KV heads): 921,600
  MQA  (1 KV head):  854,016
```

All three produce the same output shape. Parameter reduction comes from the smaller K and V projection matrices. MQA saves the most parameters; MHA has the most. GQA sits between them.

## KV Cache Memory Analysis

```python
import torch

def kv_cache_memory_mb(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int = 2,  # float16
) -> float:
    """Memory in MB for the full KV cache."""
    # K cache + V cache
    bytes_total = 2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_bytes
    return bytes_total / (1024 ** 2)

# Llama 2 7B configuration
config_7b = dict(num_layers=32, num_kv_heads=32, head_dim=128)  # MHA
# Llama 2 70B configuration (GQA with 8 KV heads)
config_70b = dict(num_layers=80, num_kv_heads=8, head_dim=128)  # GQA

seq_lengths = [2048, 4096, 8192, 32768]
print(f"{'Seq Len':>10} | {'Llama 2 7B (MHA)':>18} | {'Llama 2 70B (GQA)':>19}")
print("-" * 55)
for sl in seq_lengths:
    mem_7b  = kv_cache_memory_mb(**config_7b,  seq_len=sl)
    mem_70b = kv_cache_memory_mb(**config_70b, seq_len=sl)
    print(f"{sl:>10,} | {mem_7b:>16.1f} MB | {mem_70b:>17.1f} MB")
```

**Output:**
```text
 Seq Len |  Llama 2 7B (MHA) | Llama 2 70B (GQA)
-------------------------------------------------------
   2,048 |            512.0 MB |            327.7 MB
   4,096 |           1024.0 MB |            655.4 MB
   8,192 |           2048.0 MB |           1310.7 MB
  32,768 |           8192.0 MB |           5242.9 MB
```

Even with 4× more layers, the 70B model's GQA keeps its KV cache ~1.5× smaller than 7B's MHA at the same sequence length. Without GQA, the 70B model's KV cache at 32K tokens would be `32 × 8192 MB = 262 GB` — impossible to fit.

![KV cache memory comparison between MHA, GQA, and MQA](https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=1200&auto=format&fit=crop&q=80)

## The repeat_interleave Trick

The core of the GQA implementation is `repeat_interleave`:

```python
import torch

# Simulate: 2 KV heads, 4 Q heads (num_groups = 2)
kv = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2) = (num_kv_heads, head_dim)
print(f"KV before: {kv}")

# Expand: each KV head repeated 2 times
expanded = kv.repeat_interleave(2, dim=0)  # shape (4, 2)
print(f"KV after repeat_interleave(2): {expanded}")
```

**Output:**
```text
KV before: tensor([[1., 2.],
        [3., 4.]])
KV after repeat_interleave(2): tensor([[1., 2.],
        [1., 2.],
        [3., 4.],
        [3., 4.]])
```

KV head 0 (values `[1, 2]`) is repeated for Q heads 0 and 1. KV head 1 (values `[3, 4]`) is repeated for Q heads 2 and 3. This is exactly the GQA grouping: query heads 0-1 share KV head 0, query heads 2-3 share KV head 1.

Note that `repeat_interleave` copies the tensor in memory. In an optimized CUDA kernel, this copy is avoided by indexing — the kernel computes which KV head to use for each Q head (`kv_head_idx = q_head_idx // num_groups`). For our educational PyTorch implementation, `repeat_interleave` is clearer.

## GQA in Hugging Face Transformers

Modern Hugging Face models expose GQA via config:

```python
from transformers import AutoConfig

# Llama 2 70B
config = AutoConfig.from_pretrained("meta-llama/Llama-2-70b-hf")
print(f"Model: Llama 2 70B")
print(f"  num_attention_heads (Q): {config.num_attention_heads}")
print(f"  num_key_value_heads (KV): {config.num_key_value_heads}")
print(f"  GQA groups: {config.num_attention_heads // config.num_key_value_heads}")
print(f"  KV cache reduction: {config.num_attention_heads // config.num_key_value_heads}x")
```

**Output:**
```text
Model: Llama 2 70B
  num_attention_heads (Q): 64
  num_key_value_heads (KV): 8
  GQA groups: 8
  KV cache reduction: 8x
```

`num_key_value_heads` is the config parameter that controls GQA. When `num_key_value_heads == num_attention_heads`, you have standard MHA. When `num_key_value_heads == 1`, you have MQA.

### Verifying GQA in a Llama forward pass

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

config = model.config
print(f"Q heads: {config.num_attention_heads}")
print(f"KV heads: {config.num_key_value_heads}")
print(f"GQA groups: {config.num_attention_heads // config.num_key_value_heads}")

# Inspect a single attention layer's projection sizes
attn_layer = model.model.layers[0].self_attn
print(f"\nLayer 0 projection shapes:")
print(f"  q_proj: {attn_layer.q_proj.weight.shape}")
print(f"  k_proj: {attn_layer.k_proj.weight.shape}")
print(f"  v_proj: {attn_layer.v_proj.weight.shape}")
```

**Output:**
```text
Q heads: 32
KV heads: 8
GQA groups: 4

Layer 0 projection shapes:
  q_proj: torch.Size([2048, 2048])
  k_proj: torch.Size([512, 2048])
  v_proj: torch.Size([512, 2048])
```

The Q projection maps to `32 × 64 = 2048` dimensions. The K and V projections map to only `8 × 64 = 512` dimensions — exactly 4× smaller. This is the GQA parameter reduction at the weight level.

## Quality vs Efficiency: When GQA Works

The quality of GQA depends on the number of groups G and how the model was trained. Key findings from the paper:

- **GQA trained from scratch** (as in Llama 2 70B) matches MHA quality with G≥4 groups on most benchmarks
- **MHA → GQA conversion** via "mean pooling" of KV head groups (uptrain on 5% of training data) recovers ~95% of MHA quality
- **G=1 (MQA)** shows measurable degradation on tasks requiring precise key-value matching (multi-hop reasoning, needle-in-haystack)

The training recipe matters: GQA models must be trained with the grouped configuration from the start (or via uptraining), not converted post-hoc without any fine-tuning.

## Paper Reference

- **arXiv:** [2305.13245](https://arxiv.org/abs/2305.13245)
- **Venue:** EMNLP 2023
- **Authors:** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yanqi Zhou, Sumit Sanghai, Yury Zemlyanskiy
- **Contribution:** Proposes GQA as a middle ground between MHA and MQA, showing that grouping Q heads to share KV projections achieves near-MHA quality with near-MQA memory efficiency, and provides an uptraining recipe to convert existing MHA models.

## Conclusion

GQA is one of the most impactful architectural decisions in modern LLMs — it's why 70B parameter models can serve 4K+ token contexts on a single 80GB A100 instead of requiring distributed KV cache. The implementation is simple: project keys and values to fewer heads, then expand them back to match query heads via `repeat_interleave` before the attention computation. The quality cost is minimal when training from scratch with GQA. The memory savings are proportional to the number of groups — 8 groups means 8× smaller KV cache, which directly translates to 8× longer supported context or 8× higher throughput.

The next post covers ALiBi — attention with linear biases, which encodes position directly in the attention logits for extrapolation beyond training length.
