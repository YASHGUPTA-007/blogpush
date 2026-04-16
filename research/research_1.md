---
title: "Flash Attention 2: IO-Aware Exact Attention, Memory Math, and PyTorch Implementation"
excerpt: "Flash Attention 2 isn't approximate attention — it's exact attention implemented to minimize HBM reads/writes. Here's the memory math and a working PyTorch implementation."
author: "Soham Sharma"
category: "AI"
tags: ["Research", "Attention", "Transformers", "PyTorch", "LLMs"]
status: "published"
featuredImage: "https://images.unsplash.com/photo-1532094349884-543559b27a3d?w=1200&auto=format&fit=crop&q=80"
---

Standard attention is memory-bound, not compute-bound. The naive implementation materializes an N×N attention matrix in GPU High Bandwidth Memory (HBM), and for long sequences that memory traffic dominates total runtime — not the actual floating-point operations. Flash Attention 2 eliminates most of that HBM traffic through a tiled computation strategy, delivering exact (not approximate) attention at 2-4× the speed of the standard implementation. Understanding the mechanics gives you the foundation to reason about why sequence length is a first-class concern in LLM design.

![Researcher working with data visualizations on large monitors in a dark lab](https://images.unsplash.com/photo-1532094349884-543559b27a3d?w=1200&auto=format&fit=crop&q=80)

## The Memory Problem with Naive Attention

Standard scaled dot-product attention:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

For a sequence of length N with head dimension d:

- Q, K, V matrices: N × d each
- Attention matrix S = QK^T: N × N
- After softmax: P = softmax(S), shape N × N
- Output O = PV: N × d

The N × N matrix is the problem. For N = 8192 and float16:

```
Memory for S = 8192 × 8192 × 2 bytes = 134 MB per head
A 72-layer, 8-head model: 134 MB × 8 × 72 = 77 GB just for attention matrices
```

That's before you store P, the intermediate softmax output. And each forward pass reads and writes these matrices from HBM, which on an A100 has ~2 TB/s bandwidth. The bottleneck is the memory transfers, not the FLOPs.

## Flash Attention's Core Insight: Tiling + Online Softmax

Flash Attention avoids ever materializing the full N × N matrix. Instead, it computes attention in **tiles** — small blocks of the Q, K, V matrices that fit in SRAM (the on-chip cache, ~20 MB on A100). The challenge: softmax requires knowing the row maximum across all N elements before you can compute the normalized values. You can't finalize any output until you've seen all keys.

The solution is the **online softmax algorithm** — track a running maximum and rescale previous partial results as you see more of the row:

```python
import torch
import math

def online_softmax_demo(scores_row: torch.Tensor) -> torch.Tensor:
    """
    Demonstrates online softmax — computes softmax in one pass
    without materializing all scores first.
    
    Args:
        scores_row: 1D tensor of raw attention scores for one query
    """
    running_max = float('-inf')
    running_sum = 0.0
    values = []
    
    # First pass: compute max and exp-sum incrementally
    for score in scores_row:
        score = score.item()
        new_max = max(running_max, score)
        # Rescale previous sum with new max
        running_sum = running_sum * math.exp(running_max - new_max) + math.exp(score - new_max)
        running_max = new_max
        values.append(score)
    
    # Final normalization
    result = torch.tensor([math.exp(v - running_max) / running_sum for v in values])
    
    # Verify against torch.softmax
    reference = torch.softmax(scores_row, dim=0)
    assert torch.allclose(result, reference, atol=1e-5), "Online softmax mismatch"
    return result

scores = torch.randn(128)
output = online_softmax_demo(scores)
print(f"Max difference from reference: {(output - torch.softmax(scores, dim=0)).abs().max():.2e}")
```

Flash Attention extends this to 2D tiles: it processes blocks of queries against blocks of keys/values, maintaining per-row statistics (max and log-sum-exp) to produce correct final outputs.

## Flash Attention 2 Improvements Over v1

Flash Attention 1 had suboptimal work partitioning — threads were assigned in ways that caused synchronization overhead. FA2 made three key improvements:

1. **Fewer non-matmul FLOPs**: Reorganized the rescaling arithmetic to reduce operations outside the core matrix multiplies (which are what tensor cores accelerate).
2. **Better parallelism over sequence length**: Distributed the outer loop (over Q tiles) across thread blocks, enabling better GPU utilization on long sequences.
3. **Improved work sharing for multi-head attention**: Each attention head processes independently, and FA2 partitions more efficiently across both batch and head dimensions.

The result: ~2× throughput improvement over FA1 for forward pass, and better memory efficiency on forward-backward for training.

## Using Flash Attention 2 in PyTorch

As of PyTorch 2.0+, Flash Attention 2 is integrated via `F.scaled_dot_product_attention` with automatic dispatch:

```python
import torch
import torch.nn.functional as F

def flash_attention_pytorch(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
    """
    Uses Flash Attention 2 when available via PyTorch's SDPA.
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        v: Value tensor [batch, heads, seq_len, head_dim]
        attn_mask: Optional attention mask
        dropout_p: Dropout probability (0.0 for inference)
        is_causal: If True, applies causal masking (autoregressive)
    """
    # PyTorch selects Flash Attention, Memory-Efficient Attention, or
    # Math (fallback) based on inputs and hardware
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,      # Disable fallback to see if FA is available
        enable_mem_efficient=True
    ):
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal
        )
    return output

# Example: Single-head attention on a long sequence
batch_size = 2
num_heads = 8
seq_len = 4096
head_dim = 64

device = torch.device("cuda")
dtype = torch.float16  # FA2 requires fp16 or bf16

q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

# Causal attention (for autoregressive LMs)
output = flash_attention_pytorch(q, k, v, is_causal=True)
print(f"Output shape: {output.shape}")  # [2, 8, 4096, 64]

# Measure memory usage
torch.cuda.reset_peak_memory_stats()
output = flash_attention_pytorch(q, k, v, is_causal=True)
peak_mem_fa = torch.cuda.max_memory_allocated() / 1e9

torch.cuda.reset_peak_memory_stats()
# Standard attention for comparison
scale = head_dim ** -0.5
attn_weights = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
output_std = attn_weights @ v
peak_mem_std = torch.cuda.max_memory_allocated() / 1e9

print(f"Flash Attention peak memory: {peak_mem_fa:.2f} GB")
print(f"Standard Attention peak memory: {peak_mem_std:.2f} GB")
print(f"Memory reduction: {peak_mem_std / peak_mem_fa:.1f}x")
```

## Manual Flash Attention in Triton (Conceptual)

The actual FA2 implementation uses Triton (a GPU kernel language). Here's a simplified NumPy version that illustrates the tiling logic — not for production use, but to cement the algorithm:

```python
import numpy as np

def flash_attention_numpy(Q, K, V, block_size=64):
    """
    Simplified Flash Attention — demonstrates tiling + online softmax.
    NOT numerically optimized, for illustration only.
    
    Args:
        Q: [N, d] query matrix
        K: [N, d] key matrix  
        V: [N, d] value matrix
        block_size: tile size (B in the FA paper)
    
    Returns:
        O: [N, d] output matrix
    """
    N, d = Q.shape
    scale = d ** -0.5
    
    # Output and log-sum-exp statistics
    O = np.zeros((N, d), dtype=np.float32)
    L = np.full(N, float('-inf'), dtype=np.float32)   # running log-sum-exp
    M = np.full(N, float('-inf'), dtype=np.float32)   # running max
    
    # Outer loop: iterate over K, V blocks (columns in attention matrix)
    for j_start in range(0, N, block_size):
        j_end = min(j_start + block_size, N)
        Kj = K[j_start:j_end]   # [Bj, d]
        Vj = V[j_start:j_end]   # [Bj, d]
        
        # Inner loop: iterate over Q blocks (rows in attention matrix)
        for i_start in range(0, N, block_size):
            i_end = min(i_start + block_size, N)
            Qi = Q[i_start:i_end]   # [Bi, d]
            
            # Compute this tile's scores: [Bi, Bj]
            Sij = (Qi @ Kj.T) * scale
            
            # Tile-local max and exp
            m_new = np.maximum(M[i_start:i_end], Sij.max(axis=1))   # [Bi]
            
            # Rescale previous accumulator
            rescale = np.exp(M[i_start:i_end] - m_new)   # [Bi]
            O[i_start:i_end] = O[i_start:i_end] * rescale[:, None]
            
            # Add this tile's contribution
            exp_Sij = np.exp(Sij - m_new[:, None])   # [Bi, Bj]
            O[i_start:i_end] += exp_Sij @ Vj
            
            # Update running statistics
            L[i_start:i_end] = (
                np.exp(M[i_start:i_end] - m_new) * np.exp(L[i_start:i_end]) +
                exp_Sij.sum(axis=1)
            )
            M[i_start:i_end] = m_new
    
    # Final normalization
    O = O / np.exp(L)[:, None]
    return O


# Verify correctness against standard attention
np.random.seed(42)
N, d = 128, 64
Q = np.random.randn(N, d).astype(np.float32)
K = np.random.randn(N, d).astype(np.float32)
V = np.random.randn(N, d).astype(np.float32)

fa_output = flash_attention_numpy(Q, K, V)

# Reference: standard attention
scale = d ** -0.5
scores = Q @ K.T * scale
attn = np.exp(scores - scores.max(axis=1, keepdims=True))
attn /= attn.sum(axis=1, keepdims=True)
ref_output = attn @ V

max_diff = np.abs(fa_output - ref_output).max()
print(f"Max difference from reference: {max_diff:.2e}")  # Should be < 1e-4
```

## Memory Complexity Analysis

Standard attention memory complexity: **O(N²)** for the attention matrix.
Flash Attention memory complexity: **O(N)** — the tile size is fixed, and we only keep the output and running statistics.

```
Block size B: typically 64-128 (fits in SRAM)
SRAM usage per block: 4 * B * d (Q, K, V tiles + output tile)
For B=64, d=64: 4 * 64 * 64 * 2 bytes (fp16) = 32 KB per tile
A100 SRAM: ~20 MB — can hold many tiles simultaneously
```

The key insight: as N grows from 1K to 64K tokens, standard attention memory grows 4096×, while Flash Attention memory stays constant (proportional to d and B, not N).

![Abstract visualization of memory blocks and computation tiles on GPU hardware](https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200&auto=format&fit=crop&q=80)

## Practical Considerations

**When FA2 activates automatically**: PyTorch's `F.scaled_dot_product_attention` selects FA2 when:
- Running on CUDA with compute capability ≥ 8.0 (A100, H100, etc.)
- Input dtype is float16 or bfloat16
- Head dimension ≤ 256
- No custom attention masks (or causal mask)

```python
# Check which backend PyTorch will use
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
    # PyTorch will log which kernel it selects if you enable debug logging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    _ = F.scaled_dot_product_attention(q, k, v)
```

**Custom attention masks with FA2**: FA2 doesn't support arbitrary attention masks directly (only causal). For block-sparse or sliding window attention patterns, use libraries like `xformers` or implement via the Triton kernel directly.

**Training with FA2**: FA2 maintains the correct gradient computation through a clever recomputation strategy — during the backward pass, it recomputes tiles of the attention matrix from the stored statistics (m and L) rather than storing the full N × N matrix. This is what enables sub-quadratic memory for training, not just inference.

## Paper Reference

**FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**

- **arXiv**: [2307.08691](https://arxiv.org/abs/2307.08691)
- **Venue**: ICLR 2024
- **Authors**: Tri Dao
- **Key contribution**: Rewrites the thread-block assignment of FA1 to reduce non-matmul FLOPs and improve GPU occupancy, achieving ~2× throughput improvement over FA1 and making 100K-token context windows practical.

## Conclusion

Flash Attention 2 is not a new algorithm for computing attention — it's the same mathematical operation implemented with awareness of GPU memory hierarchy. The N × N matrix still exists conceptually; FA2 just never materializes it in full. Understanding the tile + online-softmax strategy gives you the mental model to reason about when context length becomes a bottleneck, why FA2 requires specific dtypes and hardware, and how future attention variants (like ring attention for multi-node training) extend these ideas further.

The practical takeaway: any production LLM codebase should use `F.scaled_dot_product_attention` with Flash Attention enabled. It's exact, it's faster, and the memory savings compound dramatically as sequence length grows.
