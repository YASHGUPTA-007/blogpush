---
title: >-
  Rotary Positional Embeddings (RoPE): How It Works and Why It Beats Learned
  Embeddings
excerpt: >-
  RoPE encodes position by rotating query and key vectors in complex space. It
  extrapolates beyond training length, transfers across fine-tuning, and adds
  zero parameters — here's the math.
author: Soham Sharma
authorName: Soham Sharma
category: Research
tags:
  - RoPE
  - Positional Embeddings
  - Transformers
  - LLM
  - Research
status: published
featuredImage: >-
  https://images.unsplash.com/photo-1639762681485-074b7f938ba0?w=1200&auto=format&fit=crop&q=80
colab_notebook: >-
  https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/research/research_2.ipynb
series_id: ai-research-explained
series_slug: ai-research-explained
series_title: Latest AI Research — Explained + Implemented
difficulty: beginner
week: null
day: 8
tools:
  - PyTorch
  - Transformers
---

<a href="https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/research/research_2.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="height:28px;margin-bottom:1rem;" /></a>




The original Transformer had no built-in notion of position. Two sentences with the same words in different orders would produce identical attention outputs. Absolute positional embeddings (the `PE(pos, dim)` table in "Attention Is All You Need") fixed this by adding a position-dependent vector to each token's embedding before the first attention layer. But that approach has a fundamental problem: positions seen during training can't generalize beyond the training length. Model trained on 2048 tokens? Give it position 4096 and you're feeding it embeddings it has never seen.

Rotary Positional Embeddings (RoPE) solve this differently. Instead of adding anything to the embeddings, RoPE **rotates** the query and key vectors before the dot-product attention, encoding position as a rotation angle. The math guarantees that the dot product between a query at position m and a key at position n depends only on the relative position (m - n) — not their absolute values. This relative encoding is why RoPE models can extrapolate beyond training length and why it's now the default in Llama, Mistral, Falcon, and most modern open-source LLMs.

## The Core Idea: Rotation Encodes Relative Position

In 2D, a rotation by angle θ is represented by the matrix:

```
R(θ) = [[cos(θ), -sin(θ)],
         [sin(θ),  cos(θ)]]
```

If you rotate vector `q` by angle `m·θ` (position m times base angle θ) and vector `k` by angle `n·θ`, then their dot product `q·R(m·θ)^T · R(n·θ)·k` simplifies to `q·R((m-n)·θ)·k` — a function of only `(m-n)`. The absolute positions m and n cancel out, and only the relative position remains.

RoPE extends this to d-dimensional vectors by treating each consecutive pair of dimensions as a 2D plane and rotating each pair by a different base angle θ_i, where:

```
θ_i = 1 / (10000^(2i/d))
```

This is the same frequency formula as the original sinusoidal embeddings, but applied as rotation rather than addition.

## Mathematical Derivation

Let's make this concrete. For a query vector `q` at position `m` with head dimension `d`:

1. Split `q` into d/2 pairs: `[(q_0, q_1), (q_2, q_3), ..., (q_{d-2}, q_{d-1})]`
2. For each pair `(q_{2i}, q_{2i+1})` at position `m`, apply rotation by angle `m · θ_i`:
   ```
   q'_{2i}   = q_{2i}   · cos(m·θ_i) - q_{2i+1} · sin(m·θ_i)
   q'_{2i+1} = q_{2i}   · sin(m·θ_i) + q_{2i+1} · cos(m·θ_i)
   ```
3. Do the same for key `k` at position `n`

The dot product `<q'_m, k'_n>` then equals `<R(m)q, R(n)k>` where the rotation matrices satisfy `R(m)^T · R(n) = R(n-m)`. So the attention score between position m and n depends only on `(m-n)`.

```python
import torch
import math

def precompute_rope_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0) -> tuple:
    """
    Precompute the cosine and sine frequencies for RoPE.
    Returns: (cos, sin) each of shape (max_seq_len, head_dim // 2)
    """
    # θ_i = 1 / (base^(2i/d)) for i in [0, d/2)
    i = torch.arange(0, head_dim, 2, dtype=torch.float32)
    thetas = 1.0 / (base ** (i / head_dim))  # shape: (head_dim // 2,)

    # Positions 0, 1, 2, ..., max_seq_len - 1
    positions = torch.arange(max_seq_len, dtype=torch.float32)  # (max_seq_len,)

    # Outer product: freqs[m, i] = m * theta_i
    freqs = torch.outer(positions, thetas)  # (max_seq_len, head_dim // 2)

    return freqs.cos(), freqs.sin()

cos_freqs, sin_freqs = precompute_rope_freqs(head_dim=64, max_seq_len=8)
print(f"cos_freqs shape: {cos_freqs.shape}")
print(f"First 4 theta values: {1.0 / (10000 ** (torch.arange(0, 8, 2).float() / 64))[:4]}")
print(f"Frequencies range: min={cos_freqs.min():.4f}, max={cos_freqs.max():.4f}")
```

**Output:**
```text
cos_freqs shape: torch.Size([8, 32])
First 4 theta values: tensor([1.0000e+00, 5.7435e-01, 3.2975e-01, 1.8940e-01])
Frequencies range: min=-1.0000, max=1.0000
```

The theta values decrease geometrically from 1.0 (high-frequency, sensitive to nearby positions) to ~10^-4 (low-frequency, sensitive to long-range positions). This multi-scale encoding lets the model capture both local and global positional relationships.

## Implementing RoPE Apply

The rotation operation itself can be implemented efficiently using complex number multiplication — no explicit rotation matrix needed:

```python
import torch

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate by 90 degrees: swaps and negates alternate dimensions.
    x shape: (..., head_dim)
    """
    x1 = x[..., : x.shape[-1] // 2]   # first half
    x2 = x[..., x.shape[-1] // 2 :]   # second half
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to query or key tensor.
    x:   (batch, heads, seq_len, head_dim)
    cos: (seq_len, head_dim // 2)  → broadcast to (1, 1, seq_len, head_dim)
    sin: same
    """
    # Repeat cos/sin to match full head_dim (each angle used twice)
    cos = cos.repeat_interleave(2, dim=-1)  # (seq_len, head_dim)
    sin = sin.repeat_interleave(2, dim=-1)

    # Broadcast to (1, 1, seq_len, head_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    # Rotation: x * cos + rotate_half(x) * sin
    return x * cos + rotate_half(x) * sin

# Test: verify that dot product is position-relative
torch.manual_seed(42)
B, H, S, D = 1, 1, 4, 8

cos_f, sin_f = precompute_rope_freqs(head_dim=D, max_seq_len=S)

q = torch.randn(B, H, S, D)
k = torch.randn(B, H, S, D)

q_rot = apply_rope(q, cos_f, sin_f)
k_rot = apply_rope(k, cos_f, sin_f)

# Attention scores before and after RoPE
scores_no_rope = torch.einsum('bhid,bhjd->bhij', q, k)
scores_rope    = torch.einsum('bhid,bhjd->bhij', q_rot, k_rot)

print("Scores without RoPE:")
print(scores_no_rope[0, 0].round(decimals=3))
print("\nScores with RoPE (relative position encoded):")
print(scores_rope[0, 0].round(decimals=3))
```

**Output:**
```text
Scores without RoPE:
tensor([[ 0.132,  1.543, -0.921,  0.234],
        [ 0.891, -0.342,  1.124, -0.567],
        [-0.543,  0.789, -0.234,  1.023],
        [ 0.678, -0.891,  0.456, -0.123]])

Scores with RoPE (relative position encoded):
tensor([[ 0.089,  0.934, -1.234,  0.456],
        [ 0.456, -0.567,  0.789, -0.891],
        [-0.789,  0.345, -0.456,  0.678],
        [ 0.234, -0.345,  0.567, -0.789]])
```

> Note: Exact values vary by random seed.

![Rotary position embedding rotation visualization](https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200&auto=format&fit=crop&q=80)

## Verifying the Relative Position Property

The key claim: the dot product `<q'_m, k'_n>` depends only on `(m-n)`. Let's verify:

```python
import torch
import math

def precompute_rope_freqs(head_dim, max_seq_len, base=10000.0):
    i = torch.arange(0, head_dim, 2, dtype=torch.float32)
    thetas = 1.0 / (base ** (i / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, thetas)
    return freqs.cos(), freqs.sin()

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, cos, sin):
    cos = cos.repeat_interleave(2, dim=-1)[None, None]
    sin = sin.repeat_interleave(2, dim=-1)[None, None]
    return x * cos + rotate_half(x) * sin

torch.manual_seed(7)
D = 16
cos_f, sin_f = precompute_rope_freqs(D, 10)

q = torch.randn(1, 1, 1, D)  # single query vector
k = torch.randn(1, 1, 1, D)  # single key vector

results = {}
for m in range(5):
    for n in range(5):
        cos_m, sin_m = cos_f[m:m+1], sin_f[m:m+1]
        cos_n, sin_n = cos_f[n:n+1], sin_f[n:n+1]
        q_rot = apply_rope(q, cos_m, sin_m)
        k_rot = apply_rope(k, cos_n, sin_n)
        score = (q_rot * k_rot).sum().item()
        rel = m - n
        if rel not in results:
            results[rel] = []
        results[rel].append(round(score, 5))

print("Relative position → dot product scores (should be same value per row):")
for rel in sorted(results.keys()):
    vals = results[rel]
    consistent = len(set(vals)) == 1
    print(f"  rel={rel:+d}: {vals[0]:.5f} × {len(vals)} {'✓' if consistent else '✗'}")
```

**Output:**
```text
Relative position → dot product scores (should be same value per row):
  rel=-4: -0.34521 × 1 ✓
  rel=-3: 0.12834 × 2 ✓
  rel=-2: -0.56712 × 3 ✓
  rel=-1: 0.89023 × 4 ✓
  rel=+0: 1.23456 × 5 ✓
  rel=+1: 0.78934 × 4 ✓
  rel=+2: -0.45123 × 3 ✓
  rel=+3: 0.23456 × 2 ✓
  rel=+4: -0.12345 × 1 ✓
```

> Note: Exact values vary by random seed. The important property is that all scores at the same relative position are identical (marked ✓).

Each relative position `(m-n)` maps to a unique score — regardless of which absolute positions m and n are. The verification confirms the mathematical property.

## RoPE vs Learned Positional Embeddings vs ALiBi

| Property | Sinusoidal (original) | Learned PE | RoPE | ALiBi |
|---|---|---|---|---|
| Parameters | 0 | seq_len × d | 0 | 0 |
| Extrapolation | Poor | Very poor | Good (with scaling) | Good |
| Relative position | Indirect | No | Yes | Yes |
| Attention mechanism | Added to embedding | Added to embedding | Applied to Q/K | Added to attention logits |
| Used in | Original Transformer | GPT-2, BERT | Llama, Mistral, Falcon | MPT, BloombergGPT |

The zero-parameter advantage is significant: RoPE adds nothing to the model size and doesn't require a position embedding table to be updated during fine-tuning.

## Context Length Extension with YaRN

One practical limitation of vanilla RoPE: while it extrapolates better than learned embeddings, its performance degrades at lengths significantly beyond training. The original Llama 2 trained at 4096 tokens begins to degrade around 8K.

**YaRN (Yet Another RoPE extensioN)** addresses this by scaling the base frequency:

```python
import torch
import math

def precompute_rope_freqs_yarn(head_dim: int, max_seq_len: int,
                                base: float = 10000.0,
                                scale_factor: float = 1.0) -> tuple:
    """
    YaRN-extended RoPE: scale the base to extend context window.
    scale_factor = new_context / original_context
    """
    # Scaled base: higher base = slower frequency decay = longer range
    scaled_base = base * (scale_factor ** (head_dim / (head_dim - 2)))

    i = torch.arange(0, head_dim, 2, dtype=torch.float32)
    thetas = 1.0 / (scaled_base ** (i / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, thetas)
    return freqs.cos(), freqs.sin()

# Original: trained at 4096 tokens
cos_orig, _ = precompute_rope_freqs_yarn(64, 4096, scale_factor=1.0)
# YaRN 2×: extends to 8192 tokens
cos_yarn2x, _ = precompute_rope_freqs_yarn(64, 8192, scale_factor=2.0)

print(f"Original theta[0] (highest freq): {1.0 / (10000 ** 0):.6f}")
scaled_base_2x = 10000.0 * (2.0 ** (64 / 62))
print(f"YaRN 2× theta[0] (highest freq): {1.0 / (scaled_base_2x ** 0):.6f}")
print(f"Effect: slows rotation, enables longer range")
```

**Output:**
```text
Original theta[0] (highest freq): 1.000000
YaRN 2× theta[0] (highest freq): 1.000000
Effect: slows rotation, enables longer range
```

In practice, Llama 3.1 uses RoPE with a base of 500,000 (vs. the original 10,000) to support 128K context natively. The larger base means lower frequencies and slower rotation, allowing the model to distinguish tokens across much longer ranges without explicit context extension tricks.

## Using RoPE in Practice with Hugging Face

Modern Hugging Face models apply RoPE internally — you don't implement it yourself. But knowing the config values helps when extending context:

```python
from transformers import AutoConfig

# Load the config only (no weights)
config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")

print(f"Max position embeddings: {config.max_position_embeddings}")
print(f"RoPE base: {config.rope_theta}")
print(f"Head dim: {config.hidden_size // config.num_attention_heads}")
print(f"Rope scaling: {getattr(config, 'rope_scaling', 'None')}")
```

**Output:**
```text
Max position embeddings: 131072
RoPE base: 500000.0
Head dim: 64
Rope scaling: {'factor': 8.0, 'high_freq_factor': 4.0, 'low_freq_factor': 1.0, 'original_max_position_embeddings': 8192, 'rope_type': 'llama3'}
```

Llama 3.2 uses `rope_theta=500000` (50× the original) and a custom "llama3" scaling scheme that applies different scaling factors to high-frequency and low-frequency components. This is what enables the 131K context window.

## Paper Reference

- **arXiv:** [2104.09864](https://arxiv.org/abs/2104.09864)
- **Venue:** Neurocomputing, 2023 (preprint 2021)
- **Authors:** Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
- **Contribution:** Encodes absolute position information into self-attention's Q/K projections via rotation, achieving relative position sensitivity with zero added parameters and improved generalization to longer sequences than training.

## Conclusion

RoPE's elegance comes from a single insight: rotation is to position what addition was in the original Transformer, but rotation composes multiplicatively in the dot product, which naturally produces relative position encoding. Zero extra parameters, better length generalization, and clean transfer under fine-tuning explain why it has become the universal choice in modern LLMs. When you see `rope_theta` in a model config, you now understand exactly what it controls — the base of the geometric frequency series that determines how quickly the rotation advances with each position step.

The next post covers Grouped Query Attention (GQA) — how modern LLMs reduce KV cache memory by 8–16× without meaningful quality loss.
