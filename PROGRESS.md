# Blog Generation Progress

**Total: 80 posts across 4 categories**
**Pattern: Round-robin (1 post per category per round, 20 rounds)**
**Quality Standard: Enhanced — output blocks on all code, detailed explanations (see MASTER_PROMPT.md)**
**Author: Soham Sharma**

---

## Status Key
- ✅ Complete
- 🔄 In Progress
- ⏳ Pending

---

## Setup
- ✅ Created folders: `pytorch/` `langchain/` `tensorflow/` `research/`
- ✅ Created `notebooks/` folder (auto-populated by CI)
- ✅ Created `MASTER_PROMPT.md` (combined prompt file)
- ✅ Created `scripts/generate-colab.mts` (auto Colab notebook generation)
- ✅ Created `scripts/inject-colab-link.mts` (auto badge injection)
- ✅ Created `.github/workflows/generate-colabs.yml` (GitHub Actions pipeline)

---

## Rounds

| Round | pytorch | langchain | tensorflow | research | Status |
|-------|---------|-----------|------------|----------|--------|
| 1  | pytorch_1 ✅  | langchain_1 ✅  | tensorflow_1 ✅  | research_1 ✅  | ✅ Done |
| 2  | pytorch_2 ✅  | langchain_2 ✅  | tensorflow_2 ✅  | research_2 ✅  | ✅ Done |
| 3  | pytorch_3 ✅  | langchain_3 ✅  | tensorflow_3 ✅  | research_3 ✅  | ✅ Done |
| 4  | pytorch_4 ✅  | langchain_4 ✅  | tensorflow_4 ✅  | research_4 ✅  | ✅ Done |
| 5  | pytorch_5 ✅  | langchain_5 ✅  | tensorflow_5 ✅  | research_5 ✅  | ✅ Done |
| 6  | pytorch_6 ⏳  | langchain_6 ⏳  | tensorflow_6 ⏳  | research_6 ⏳  | ⏳ Pending |
| 7  | pytorch_7 ⏳  | langchain_7 ⏳  | tensorflow_7 ⏳  | research_7 ⏳  | ⏳ Pending |
| 8  | pytorch_8 ⏳  | langchain_8 ⏳  | tensorflow_8 ⏳  | research_8 ⏳  | ⏳ Pending |
| 9  | pytorch_9 ⏳  | langchain_9 ⏳  | tensorflow_9 ⏳  | research_9 ⏳  | ⏳ Pending |
| 10 | pytorch_10 ⏳ | langchain_10 ⏳ | tensorflow_10 ⏳ | research_10 ⏳ | ⏳ Pending |
| 11 | pytorch_11 ⏳ | langchain_11 ⏳ | tensorflow_11 ⏳ | research_11 ⏳ | ⏳ Pending |
| 12 | pytorch_12 ⏳ | langchain_12 ⏳ | tensorflow_12 ⏳ | research_12 ⏳ | ⏳ Pending |
| 13 | pytorch_13 ⏳ | langchain_13 ⏳ | tensorflow_13 ⏳ | research_13 ⏳ | ⏳ Pending |
| 14 | pytorch_14 ⏳ | langchain_14 ⏳ | tensorflow_14 ⏳ | research_14 ⏳ | ⏳ Pending |
| 15 | pytorch_15 ⏳ | langchain_15 ⏳ | tensorflow_15 ⏳ | research_15 ⏳ | ⏳ Pending |
| 16 | pytorch_16 ⏳ | langchain_16 ⏳ | tensorflow_16 ⏳ | research_16 ⏳ | ⏳ Pending |
| 17 | pytorch_17 ⏳ | langchain_17 ⏳ | tensorflow_17 ⏳ | research_17 ⏳ | ⏳ Pending |
| 18 | pytorch_18 ⏳ | langchain_18 ⏳ | tensorflow_18 ⏳ | research_18 ⏳ | ⏳ Pending |
| 19 | pytorch_19 ⏳ | langchain_19 ⏳ | tensorflow_19 ⏳ | research_19 ⏳ | ⏳ Pending |
| 20 | pytorch_20 ⏳ | langchain_20 ⏳ | tensorflow_20 ⏳ | research_20 ⏳ | ⏳ Pending |

---

## Files Generated: 20 / 80

### Completed Files
| # | File | Topic | Difficulty |
|---|------|-------|------------|
| 1  | pytorch/pytorch_1.md | Tensors Deep Dive — dtypes, device movement, memory layout, broadcasting | Beginner |
| 2  | langchain/langchain_1.md | LangChain Architecture Overview — chains, runnables, LCEL | Beginner |
| 3  | tensorflow/tensorflow_1.md | TensorFlow 2.x Architecture — eager execution, tf.function, AutoGraph | Beginner |
| 4  | research/research_1.md | Flash Attention 2 — IO-aware exact attention, memory math, PyTorch implementation | Beginner |
| 5  | pytorch/pytorch_2.md | Autograd Internals — computation graphs, retain_graph, grad_fn chain, detach | Beginner |
| 6  | langchain/langchain_2.md | Prompt Templates and Output Parsers — PromptTemplate, ChatPromptTemplate, Pydantic | Beginner |
| 7  | tensorflow/tensorflow_2.md | Keras Sequential vs Functional vs Subclassing — when to use which API | Beginner |
| 8  | research/research_2.md | Rotary Positional Embeddings (RoPE) — how it works, why it beats learned embeddings | Beginner |
| 9  | pytorch/pytorch_3.md | Custom Dataset and DataLoader — __getitem__, __len__, collate_fn, num_workers | Beginner |
| 10 | langchain/langchain_3.md | Working with LLMs and Chat Models — OpenAI, Anthropic, local models via Ollama | Beginner |
| 11 | tensorflow/tensorflow_3.md | Custom Training Loops with GradientTape — manual forward/backward | Beginner |
| 12 | research/research_3.md | Grouped Query Attention (GQA) — KV cache reduction, Llama 2 implementation | Beginner |
| 13 | pytorch/pytorch_4.md | Training Loop Anatomy — forward, loss, backward, optimizer.step, zero_grad | Beginner |
| 14 | langchain/langchain_4.md | Document Loaders and Text Splitters — PDF, web, recursive character splitter | Beginner |
| 15 | tensorflow/tensorflow_4.md | Data Pipelines with tf.data — map, batch, prefetch, cache, shuffle | Beginner |
| 16 | research/research_4.md | ALiBi — attention with linear biases, extrapolation beyond training length | Beginner |
| 17 | pytorch/pytorch_5.md | Building a CNN from Scratch — conv layers, pooling, BatchNorm, CIFAR-10 | Intermediate |
| 18 | langchain/langchain_5.md | Embeddings and Vector Stores — OpenAI embeddings, FAISS, Chroma, similarity search | Intermediate |
| 19 | tensorflow/tensorflow_5.md | Keras Functional API — multi-input/output models, shared layers, branching | Intermediate |
| 20 | research/research_5.md | LoRA — low-rank decomposition math, PEFT library, training on a real dataset | Intermediate |
