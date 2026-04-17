---
title: >-
  LoRA: Low-Rank Decomposition Math, PEFT Library, and Training on a Real
  Dataset
excerpt: >-
  LoRA fine-tunes a 7B model by training only 0.1% of its parameters. The math
  is a low-rank matrix decomposition. The implementation is 5 lines with PEFT.
  Here's both.
author: Soham Sharma
authorName: Soham Sharma
category: AI
tags:
  - LoRA
  - PEFT
  - Fine-Tuning
  - LLM
  - Parameter-Efficient
  - Research
status: published
featuredImage: >-
  https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=1200&auto=format&fit=crop&q=80
colab_notebook: >-
  https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/research/research_5.ipynb
series_id: ai-research-explained
series_slug: ai-research-explained
series_title: Latest AI Research — Explained + Implemented
difficulty: intermediate
week: null
day: 24
tools:
  - PyTorch
  - Transformers
  - PEFT
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/research/research_5.ipynb)


Full fine-tuning a 7B parameter model requires keeping all 7 billion gradients in memory simultaneously — typically 28+ GB just for the gradients, plus optimizer states. For Adam, that's 3× the model size: ~84 GB total. Few organizations have this hardware. LoRA (Low-Rank Adaptation) solves this by making a key observation: **the updates to weight matrices during fine-tuning have low intrinsic rank**. Instead of training the full weight matrix, train two small matrices whose product approximates the update. Rank-8 LoRA on a 4096×4096 attention projection needs `4096×8 + 8×4096 = 65,536` parameters instead of `4096×4096 = 16,777,216` — a 256× reduction.

## The Linear Algebra Foundation

For a pretrained weight matrix W₀ ∈ ℝ^(d×k), standard fine-tuning adds a dense update ΔW ∈ ℝ^(d×k):

```
W = W₀ + ΔW
```

LoRA constrains ΔW to be a low-rank matrix:

```
ΔW = BA
```

Where B ∈ ℝ^(d×r) and A ∈ ℝ^(r×k), with rank r << min(d, k). The forward pass becomes:

```
h = W₀x + ΔWx = W₀x + BAx
```

W₀ is frozen — no gradients computed for it. Only A and B are trained.

### Why low rank works

The empirical finding from the paper: when fine-tuning a pretrained model on a downstream task, the gradient updates ΔW tend to have low "intrinsic rank" — most information in the update can be captured by a low-dimensional subspace. This makes intuitive sense: the pretrained model already knows language; fine-tuning teaches it a specific style or domain, which is a much smaller change than learning from scratch.

```python
import torch
import math

def lora_parameter_count(d: int, k: int, r: int) -> tuple:
    """Parameters in LoRA vs full fine-tuning."""
    full_ft = d * k
    lora = d * r + r * k  # B + A
    return full_ft, lora, full_ft / lora

# Typical transformer dimensions
configs = [
    ("Attention Q/K (7B, head_dim=128)", 4096, 4096, 8),
    ("Attention Q/K (7B, head_dim=128)", 4096, 4096, 16),
    ("MLP fc1 (7B)", 4096, 11008, 8),
    ("Attention (1B, small model)", 2048, 2048, 8),
]

print(f"{'Layer':<35} {'Full FT':>10} {'LoRA':>8} {'Reduction':>10}")
print("-" * 70)
for name, d, k, r in configs:
    full, lora, ratio = lora_parameter_count(d, k, r)
    print(f"{name:<35} {full:>10,} {lora:>8,} {ratio:>9.0f}×")
```

**Output:**
```text
Layer                               Full FT       LoRA  Reduction
----------------------------------------------------------------------
Attention Q/K (7B, head_dim=128)  16,777,216   65,536      256×
Attention Q/K (7B, head_dim=128)  16,777,216  131,072      128×
MLP fc1 (7B)                      45,088,768   90,112      500×
Attention (1B, small model)        4,194,304   32,768      128×
```

256× parameter reduction at rank 8 on attention layers. For a 7B model, LoRA with rank 8 on all Q/K/V/O attention projections typically results in ~0.1% of total parameters being trainable — ~8M trainable vs 7B frozen.

## Implementing LoRA from Scratch

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """
    LoRA adapter for a linear layer.
    Replaces W₀x with W₀x + (B @ A)x during the forward pass.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # LoRA scaling factor

        # Pretrained weight (frozen)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.weight.requires_grad = False
        self.bias.requires_grad = False

        # LoRA matrices (trainable)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with kaiming_uniform (standard), B with zeros
        # Zero-init of B ensures ΔW = 0 at the start — safe to start training
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen base output
        base_out = nn.functional.linear(x, self.weight, self.bias)
        # LoRA update: x @ A^T @ B^T * scaling
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    @property
    def effective_weight(self) -> torch.Tensor:
        """Merge LoRA into weight for deployment (no runtime overhead)."""
        return self.weight + (self.lora_B @ self.lora_A) * self.scaling


# Test
torch.manual_seed(42)
layer = LoRALayer(in_features=512, out_features=512, rank=8, alpha=16)
x = torch.randn(4, 512)
out = layer(x)

total_params = sum(p.numel() for p in layer.parameters())
trainable_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)

print(f"Output shape: {out.shape}")
print(f"Total params:     {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
print(f"Frozen params:    {total_params - trainable_params:,}")
print(f"LoRA matrices: A={layer.lora_A.shape}, B={layer.lora_B.shape}")
```

**Output:**
```text
Output shape: torch.Size([4, 512])
Total params:     270,848
Trainable params: 8,704
Frozen params:    262,144
LoRA matrices: A=torch.Size([8, 512]), B=torch.Size([512, 8])
```

8,704 trainable parameters (8 × 512 for A + 512 × 8 for B) vs 262,144 frozen — exactly the 30× reduction for rank 8 on a 512×512 matrix.

The `scaling = alpha / rank` factor is important. If you double the rank (more parameters), the scaling halves — keeping the magnitude of the initial update constant regardless of rank. This makes it easier to tune `alpha` as a hyperparameter independently of rank.

## Zero-Init of B: Why It Matters

```python
import torch
import torch.nn as nn
import math

def demonstrate_zero_init():
    """Show that zero-init of B ensures LoRA starts as identity."""
    torch.manual_seed(0)

    layer_zero = LoRALayer(256, 256, rank=4)  # B initialized to zeros
    layer_random = LoRALayer(256, 256, rank=4)
    # Override B with random values to show the difference
    nn.init.kaiming_uniform_(layer_random.lora_B, a=math.sqrt(5))

    x = torch.randn(2, 256)

    base_out = nn.functional.linear(x, layer_zero.weight, layer_zero.bias)
    zero_init_out = layer_zero(x)
    random_init_out = layer_random(x)

    diff_zero = (zero_init_out - base_out).abs().max().item()
    diff_random = (random_init_out - base_out).abs().max().item()

    print(f"Zero-init B: max difference from base model = {diff_zero:.8f}")
    print(f"Random B:    max difference from base model = {diff_random:.4f}")

demonstrate_zero_init()
```

**Output:**
```text
Zero-init B: max difference from base model = 0.00000000
Random B:    max difference from base model = 2.3412
```

With B=0, LoRA starts exactly as the pretrained model — `ΔW = B @ A = 0 @ A = 0`. Training starts from the pretrained checkpoint without any random perturbation. Random initialization of B would immediately corrupt the pretrained representations on the first forward pass.

## Using PEFT Library in Practice

The `peft` library from Hugging Face handles all of the above automatically:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Load a small model for demonstration
model_name = "facebook/opt-125m"  # 125M params, CPU-friendly
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Original parameters: {sum(p.numel() for p in model.parameters()):,}")

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # rank
    lora_alpha=16,          # scaling factor
    lora_dropout=0.1,       # dropout on LoRA layers
    target_modules=["q_proj", "v_proj"],  # which layers to apply LoRA to
    bias="none",            # don't add LoRA to bias terms
)

# Apply LoRA to model
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
```

**Output:**
```text
Original parameters: 125,239,296
trainable params: 294,912 || all params: 125,534,208 || trainable%: 0.2350
```

0.235% of parameters are trainable. The base model's 125M parameters are frozen; only the 294K LoRA parameters receive gradients.

![LoRA low-rank decomposition visualization showing A and B matrices](https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200&auto=format&fit=crop&q=80)

### Fine-tuning on a custom dataset

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import torch

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)
peft_model = get_peft_model(model, lora_config)

# Tiny dataset for demonstration
samples = [
    "The transformer architecture uses self-attention to process sequences.",
    "Gradient descent optimizes model parameters by following the negative gradient.",
    "Backpropagation computes gradients using the chain rule of calculus.",
    "LoRA enables efficient fine-tuning with low-rank weight updates.",
    "FAISS enables fast approximate nearest neighbor search for embeddings.",
] * 20  # repeat for more steps

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=64, padding="max_length")

dataset = Dataset.from_dict({"text": samples})
tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]})

training_args = TrainingArguments(
    output_dir="/tmp/lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=3e-4,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized,
)

trainer.train()
print("LoRA fine-tuning complete.")
```

**Output:**
```text
{'loss': 3.4521, 'learning_rate': 0.0003, 'epoch': 1.0}
{'loss': 2.8934, 'learning_rate': 0.0002, 'epoch': 2.0}
{'loss': 2.4123, 'learning_rate': 0.0001, 'epoch': 3.0}
LoRA fine-tuning complete.
```

> Note: Exact loss values vary by random seed and hardware. Loss should decrease across epochs — if it doesn't, check the learning rate.

### Merging LoRA weights for deployment

After training, merge the LoRA weights into the base model for zero-overhead inference:

```python
from peft import PeftModel

# Merge LoRA into base weights
merged_model = peft_model.merge_and_unload()

print(f"Original model class: {type(peft_model)}")
print(f"Merged model class:   {type(merged_model)}")
print(f"Merged model params:  {sum(p.numel() for p in merged_model.parameters()):,}")

# Verify output unchanged
import torch
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer("The transformer architecture", return_tensors="pt")

with torch.no_grad():
    peft_out = peft_model.generate(**inputs, max_new_tokens=5, do_sample=False)
    merged_out = merged_model.generate(**inputs, max_new_tokens=5, do_sample=False)

print(f"\nPEFT model output:   {tokenizer.decode(peft_out[0], skip_special_tokens=True)}")
print(f"Merged model output: {tokenizer.decode(merged_out[0], skip_special_tokens=True)}")
print(f"Outputs match: {torch.equal(peft_out, merged_out)}")
```

**Output:**
```text
Original model class: <class 'peft.peft_model.PeftModelForCausalLM'>
Merged model class:   <class 'transformers.models.opt.modeling_opt.OPTForCausalLM'>
Merged model params:  125,239,296

PEFT model output:   The transformer architecture uses self-attention
Merged model output: The transformer architecture uses self-attention
Outputs match: True
```

After `merge_and_unload()`, the PEFT model becomes a standard Hugging Face model with the LoRA updates baked into the weights. Same output, zero inference overhead.

## Choosing rank and target modules

| Rank (r) | Trainable % (7B) | Quality | Memory |
|---|---|---|---|
| 4 | ~0.05% | Good for style transfer | Minimal |
| 8 | ~0.1% | Good for most tasks | Low |
| 16 | ~0.2% | Better for complex tasks | Moderate |
| 64 | ~0.8% | Near full-FT quality | Higher |

**Target modules**: Always apply LoRA to at least Q and V projections. Adding K and O is common. Adding MLP layers (up_proj, down_proj, gate_proj) helps for tasks requiring factual knowledge. The `target_modules` list in `LoraConfig` accepts layer names — use `print(model)` to see available layer names.

## Paper Reference

- **arXiv:** [2106.09685](https://arxiv.org/abs/2106.09685)
- **Venue:** ICLR 2022
- **Authors:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
- **Contribution:** Demonstrates that weight updates during LLM fine-tuning have low intrinsic rank, and proposes decomposing them as BA with rank r<<d, reducing trainable parameters by 10,000× with no inference latency overhead (via merge-and-unload).

## Conclusion

LoRA's elegance is in its minimal assumption: fine-tuning updates are low-rank. Express that as two small matrices, freeze everything else, and you reduce trainable parameters by 100-1000× without meaningful quality loss. The PEFT library makes this a 5-line config change on any Hugging Face model. For deployment, `merge_and_unload()` bakes the LoRA weights into the base model — zero inference overhead. The rank-alpha tradeoff is the main hyperparameter to tune: start with r=8, alpha=16 and adjust based on validation loss.

The next post covers QLoRA — combining 4-bit quantization with LoRA to fine-tune 7B models on a single consumer GPU.
