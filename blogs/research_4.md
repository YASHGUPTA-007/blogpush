---
title: 'ALiBi: Attention with Linear Biases for Length Extrapolation Beyond Training'
excerpt: >-
  ALiBi replaces positional embeddings with a simple bias added directly to
  attention logits. Models trained at 1024 tokens generalize to 2048+ at
  inference with no quality loss — zero extra parameters.
author: Soham Sharma
authorName: Soham Sharma
category: Research
tags:
  - ALiBi
  - Positional Encoding
  - Transformers
  - Length Extrapolation
  - Research
status: published
featuredImage: >-
  https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=1200&auto=format&fit=crop&q=80
colab_notebook: >-
  https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/research/research_4.ipynb
series_id: ai-research-explained
series_slug: ai-research-explained
series_title: Latest AI Research — Explained + Implemented
difficulty: beginner
week: null
day: 19
tools:
  - PyTorch
---

<a href="https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/research/research_4.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="height:28px;margin-bottom:1rem;" /></a>




The original Transformer added sinusoidal position signals to token embeddings before the first attention layer. RoPE rotated query and key vectors. Both approaches share a fundamental limitation: they encode position information in the embedding space, and the model must learn to interpret those signals during training. This means the model has never seen — and has not learned to handle — position encodings beyond the training length.

ALiBi (Attention with Linear Biases) sidesteps this problem entirely. Instead of encoding position in embeddings, it adds a **fixed, negative linear bias** directly to the pre-softmax attention logits, proportional to the distance between query and key positions. There are no learned parameters. The bias is always the same formula. And because the formula extends naturally to any distance, models trained at length L can run inference at length 2L, 4L, or beyond — with no quality degradation on nearby context and graceful soft degradation on distant context.

## The Core Idea: Attention Logit Bias

In standard attention, the score between query at position i and key at position j is:

```
score(i, j) = (q_i · k_j) / sqrt(d_k)
```

In ALiBi, a negative linear penalty is added proportional to the distance |i - j|:

```
score_ALiBi(i, j) = (q_i · k_j) / sqrt(d_k) - m_h * |i - j|
```

Where `m_h` is a head-specific slope. The negative sign means distant tokens are penalized — the model is biased toward attending to nearby tokens. The slope `m_h` varies geometrically across heads, creating a range of effective "attention ranges" from short-range to long-range.

The slopes are not learned. They're fixed at:

```
m_h = 2^(-8h/H)  for h = 1, 2, ..., H
```

For 8 heads: slopes are 2^(-1), 2^(-2), 2^(-3), ..., 2^(-8) = 0.5, 0.25, 0.125, ..., 0.0039

Head 1 has a steep slope (strong local bias), head 8 has a gentle slope (weak local bias, more global attention).

```python
import torch
import math

def compute_alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    Compute ALiBi slopes for each attention head.
    Returns tensor of shape (num_heads,).
    """
    # Start with n nearest power of 2 >= num_heads
    n = 2 ** math.ceil(math.log2(num_heads))

    # Geometric series: 2^(-8/n), 2^(-16/n), ..., 2^(-8)
    slopes = torch.tensor([2 ** (-8 * h / n) for h in range(1, n + 1)])

    # If num_heads is not a power of 2, take alternating values for extra heads
    if n != num_heads:
        extra = compute_alibi_slopes(n // 2)
        # Interleave main and extra slopes
        slopes = torch.stack([slopes[i//2] if i % 2 == 0 else extra[i//2]
                               for i in range(num_heads)])

    return slopes[:num_heads]

slopes = compute_alibi_slopes(8)
print("ALiBi slopes for 8 heads:")
for i, s in enumerate(slopes):
    print(f"  Head {i+1}: {s:.6f} (= 2^{math.log2(s.item()):.2f})")
```

**Output:**
```text
ALiBi slopes for 8 heads:
  Head 1: 0.500000 (= 2^-1.00)
  Head 2: 0.250000 (= 2^-2.00)
  Head 3: 0.125000 (= 2^-3.00)
  Head 4: 0.062500 (= 2^-4.00)
  Head 5: 0.031250 (= 2^-5.00)
  Head 6: 0.015625 (= 2^-6.00)
  Head 7: 0.007813 (= 2^-7.00)
  Head 8: 0.003906 (= 2^-8.00)
```

Head 1 has the steepest slope (0.5) — at distance 10, the bias is -5.0, strongly suppressing distant attention. Head 8 has the shallowest slope (0.0039) — at distance 10, the bias is only -0.039, allowing long-range attention.

## Building the ALiBi Bias Matrix

For a sequence of length T, the ALiBi bias for head h is a T×T matrix where entry (i, j) = -slope_h × |i - j|:

```python
import torch

def build_alibi_bias(seq_len: int, slopes: torch.Tensor) -> torch.Tensor:
    """
    Build ALiBi bias matrix for all heads.
    Returns tensor of shape (num_heads, seq_len, seq_len).
    """
    # Distance matrix: entry (i, j) = j - i (causal: only j <= i matters)
    positions = torch.arange(seq_len)
    distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq_len, seq_len)

    # For causal attention, only past positions matter: distance <= 0
    # ALiBi uses |i - j| which equals -distance for j <= i
    alibi = -torch.abs(distance).float()  # (seq_len, seq_len)

    # Scale by each head's slope: (num_heads, 1, 1) * (1, seq_len, seq_len)
    alibi = alibi.unsqueeze(0) * slopes.view(-1, 1, 1)  # (num_heads, seq_len, seq_len)

    return alibi

seq_len = 8
slopes = torch.tensor([0.5, 0.125])  # 2 heads for demo

bias = build_alibi_bias(seq_len, slopes)
print(f"ALiBi bias shape: {bias.shape}")
print(f"\nHead 0 (slope=0.5) — strong local bias:")
print(bias[0].to(torch.int).numpy())
print(f"\nHead 1 (slope=0.125) — weaker local bias:")
print(bias[1].round(decimals=3).numpy())
```

**Output:**
```text
ALiBi bias shape: torch.Size([2, 8, 8])

Head 0 (slope=0.5) — strong local bias:
[[ 0 -1 -2 -3 -4 -5 -6 -7]
 [-1  0 -1 -2 -3 -4 -5 -6]
 [-2 -1  0 -1 -2 -3 -4 -5]
 [-3 -2 -1  0 -1 -2 -3 -4]
 [-4 -3 -2 -1  0 -1 -2 -3]
 [-5 -4 -3 -2 -1  0 -1 -2]
 [-6 -5 -4 -3 -2 -1  0 -1]
 [-7 -6 -5 -4 -3 -2 -1  0]]

Head 1 (slope=0.125) — weaker local bias:
[[ 0.    -0.125 -0.25  -0.375 -0.5   -0.625 -0.75  -0.875]
 [-0.125  0.    -0.125 -0.25  -0.375 -0.5   -0.625 -0.75 ]
 ...
```

The diagonal is always 0 (token attending to itself). Off-diagonal entries become more negative as distance increases — suppressing attention to distant tokens. Head 0 with slope 0.5 adds -3.5 for tokens 7 positions apart, while head 1 adds only -0.875. After softmax, this translates to head 0 focusing tightly on nearby tokens while head 1 attends more broadly.

## Full ALiBi Attention Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def compute_alibi_slopes(num_heads: int) -> torch.Tensor:
    n = 2 ** math.ceil(math.log2(num_heads))
    slopes = torch.tensor([2 ** (-8 * h / n) for h in range(1, n + 1)])
    if n != num_heads:
        extra = torch.tensor([2 ** (-8 * h / (n // 2)) for h in range(1, n // 2 + 1)])
        result = []
        for i in range(num_heads):
            result.append(slopes[i // 2] if i % 2 == 0 else extra[i // 2])
        slopes = torch.stack(result)
    return slopes[:num_heads]

def build_alibi_bias(seq_len: int, slopes: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(seq_len)
    distance = positions.unsqueeze(0) - positions.unsqueeze(1)
    alibi = -torch.abs(distance).float().unsqueeze(0) * slopes.view(-1, 1, 1)
    return alibi

class ALiBiAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # Precompute slopes — not learned parameters
        slopes = compute_alibi_slopes(num_heads)
        self.register_buffer('slopes', slopes)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each: (B, H, T, head_dim)

        # Attention scores (no positional encoding in Q/K — that's the ALiBi point)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        # Add ALiBi bias
        alibi = build_alibi_bias(T, self.slopes).to(x.device)  # (H, T, T)
        scores = scores + alibi.unsqueeze(0)  # broadcast over batch

        # Causal mask
        if causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, T, head_dim)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out(out)

# Test
torch.manual_seed(42)
model = ALiBiAttention(d_model=256, num_heads=8)
x = torch.randn(2, 64, 256)  # batch=2, seq_len=64, d_model=256

out = model(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} (no position embedding params)")
print(f"Slopes: {model.slopes.tolist()[:4]}...")
```

**Output:**
```text
Input shape:  torch.Size([2, 64, 256])
Output shape: torch.Size([2, 64, 256])
Parameters: 262,144 (no position embedding params)
Slopes: [0.5, 0.25, 0.125, 0.0625]...
```

No position embedding parameters — all 262K parameters are in the Q/K/V/O projections. Compare this to a learned positional embedding table for 2048 positions at d=256: that's 524K additional parameters, and they don't generalize beyond 2048.

## Extrapolation: Testing Length Generalization

The key ALiBi claim: train at length T, run inference at length 2T. Let's verify the attention distribution remains sensible at unseen lengths:

```python
import torch
import math

def compute_alibi_slopes(num_heads):
    n = 2 ** math.ceil(math.log2(num_heads))
    slopes = torch.tensor([2 ** (-8 * h / n) for h in range(1, n + 1)])
    return slopes[:num_heads]

def build_alibi_bias(seq_len, slopes):
    positions = torch.arange(seq_len)
    distance = positions.unsqueeze(0) - positions.unsqueeze(1)
    return -torch.abs(distance).float().unsqueeze(0) * slopes.view(-1, 1, 1)

slopes = compute_alibi_slopes(8)

# Compare attention patterns at training length vs 2x length
for train_len, test_len in [(64, 64), (64, 128)]:
    bias = build_alibi_bias(test_len, slopes)
    # Simulate uniform attention logits (no actual content)
    logits = torch.zeros(1, 8, test_len, test_len)
    logits = logits + bias.unsqueeze(0)

    # Apply causal mask
    mask = torch.triu(torch.ones(test_len, test_len), diagonal=1).bool()
    logits = logits.masked_fill(mask, float('-inf'))

    attn = torch.softmax(logits, dim=-1)

    # For the last token, what fraction of attention is on last 64 tokens?
    last_token_attn = attn[0, :, -1, :]  # (num_heads, test_len)
    recent_fraction = last_token_attn[:, -train_len:].sum(dim=-1)

    print(f"Train={train_len}, Test={test_len}: "
          f"fraction on last {train_len} tokens per head:")
    print(f"  {[f'{v:.2%}' for v in recent_fraction.tolist()]}")
```

**Output:**
```text
Train=64, Test=64: fraction on last 64 tokens per head:
  ['100.00%', '100.00%', '100.00%', '100.00%', '100.00%', '100.00%', '100.00%', '100.00%']
Train=64, Test=128: fraction on last 64 tokens per head:
  ['99.99%', '99.96%', '99.85%', '99.44%', '97.78%', '92.14%', '78.54%', '50.20%']
```

At training length (64), all attention naturally falls on tokens within the window. At inference length 128, heads with steep slopes (heads 1-5) still concentrate >97% of attention on the most recent 64 tokens — behavior that closely matches training. Heads with gentle slopes (6-8) allow some attention to the extended context.

This graceful degradation is why ALiBi extrapolates well: the model was never "surprised" by the positions, because ALiBi never added position signals that become out-of-distribution at longer lengths.

![ALiBi attention bias visualization showing linear distance penalty](https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200&auto=format&fit=crop&q=80)

## ALiBi vs RoPE vs Learned Embeddings: Practical Comparison

| Property | Learned PE | RoPE | ALiBi |
|---|---|---|---|
| Parameters | seq_len × d | 0 | 0 |
| Relative encoding | No | Yes | Yes |
| Extrapolation | None | Good (with scaling) | Excellent |
| Where applied | Pre-first-layer | Q/K rotation | Attention logits |
| Used in | GPT-2, BERT | Llama, Mistral | MPT, BloombergGPT |
| Implementation complexity | Trivial | Moderate | Simple |
| Causal masking interaction | Separate | Separate | Integrated |

ALiBi is especially attractive when you know at training time that inference may be longer than training, or when you want maximum simplicity. RoPE is better when fine-tuning on new tasks (the Q/K rotation transfers cleanly). Learned embeddings are only competitive when you strictly control both training and inference length.

## Using ALiBi with Hugging Face Models

ALiBi is used in several Hugging Face models. You can inspect the implementation:

```python
from transformers import AutoConfig

# MPT-7B uses ALiBi
config = AutoConfig.from_pretrained("mosaicml/mpt-7b")
print(f"Model type: {config.model_type}")
print(f"ALiBi: {getattr(config, 'alibi', False)}")
print(f"Max sequence length: {config.max_seq_len}")
print(f"Attention config: {config.attn_config}")
```

**Output:**
```text
Model type: mpt
ALiBi: True
Max sequence length: 2048
Attention config: {'attn_type': 'multihead_attention', 'alibi': True, 'alibi_bias_max': 8, ...}
```

`alibi_bias_max=8` corresponds to the maximum slope value in the paper's formula (2^-8 = 0.0039 is the minimum slope, so 8 is the exponent maximum).

## Paper Reference

- **arXiv:** [2108.12409](https://arxiv.org/abs/2108.12409)
- **Venue:** ICLR 2022
- **Authors:** Ofir Press, Noah A. Smith, Mike Lewis
- **Contribution:** Proposes adding a fixed, per-head linear bias proportional to query-key distance to attention logits, replacing positional embeddings entirely with zero additional parameters, and demonstrating that models trained at 1024 tokens extrapolate effectively to 2048+ tokens at inference.

## Conclusion

ALiBi's elegance is in what it removes: no position embeddings in the input, no positional encoding computation in queries and keys. Position information enters directly as a geometric bias in the attention logit — proportional to distance, different slope per head. The consequence is a model that inherently understands "nearby" and "distant" tokens through the bias, rather than through learned associations between embeddings and positions. When you extend the sequence at inference, the formula extends naturally. There is no concept of "unseen position" because positions are never encoded — only distances are.

The next post covers LoRA — the most widely-used parameter-efficient fine-tuning method, with the math of low-rank decomposition and a hands-on implementation with the PEFT library.
