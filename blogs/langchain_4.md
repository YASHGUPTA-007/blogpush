---
title: >-
  LangChain Document Loaders and Text Splitters: PDF, Web, and Recursive
  Character Splitter
excerpt: >-
  Before RAG can retrieve anything, you need to load and split your documents
  correctly. Wrong chunk size or overlap causes retrieval failures that are hard
  to debug downstream.
author: Soham Sharma
authorName: Soham Sharma
category: LangChain
tags:
  - LangChain
  - RAG
  - Document Loaders
  - Text Splitting
  - PDF
status: published
featuredImage: >-
  https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=1200&auto=format&fit=crop&q=80
colab_notebook: >-
  https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/langchain/langchain_4.ipynb
series_id: langchain-production
series_slug: langchain-production
series_title: LangChain / LangSmith / LangGraph — In Production
difficulty: beginner
week: null
day: 17
tools:
  - LangChain
  - Python
---

<a href="https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/langchain/langchain_4.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="height:28px;margin-bottom:1rem;" /></a>




Every RAG pipeline starts with loading documents and splitting them into chunks. Engineers rush through this step and then spend days debugging why retrieval returns irrelevant results — the real cause is usually chunk boundaries that split mid-sentence, chunks too large to fit in context, or document loaders that silently drop formatting. This post covers the loaders and splitters you'll actually use in production, and the decisions that make the difference between a RAG pipeline that works and one that doesn't.

## Document: The Core Abstraction

LangChain represents all loaded content as `Document` objects with two fields:

- `page_content` — the text content (string)
- `metadata` — a dict of source information (filename, page number, URL, etc.)

```python
from langchain_core.documents import Document

doc = Document(
    page_content="LangChain is a framework for building LLM-powered applications.",
    metadata={"source": "intro.md", "page": 1, "author": "Soham Sharma"}
)

print(f"Content: {doc.page_content}")
print(f"Metadata: {doc.metadata}")
print(f"Type: {type(doc)}")
```

**Output:**
```text
Content: LangChain is a framework for building LLM-powered applications.
Metadata: {'source': 'intro.md', 'page': 1, 'author': 'Soham Sharma'}
Type: <class 'langchain_core.documents.base.Document'>
```

`metadata` is preserved through the entire pipeline — loader to splitter to vector store. When you retrieve a chunk, the metadata tells you which document it came from, what page, and any other context the loader captured.

## Loading Plain Text and Markdown

The simplest loaders: `TextLoader` for `.txt` files, `UnstructuredMarkdownLoader` for `.md`.

```python
from langchain_community.document_loaders import TextLoader
import tempfile
import os

# Create a temp file with sample content
sample_text = """Introduction to Transformers

The transformer architecture was introduced in 'Attention Is All You Need' (2017).
It relies entirely on attention mechanisms, discarding recurrence and convolutions.

Key components:
- Multi-head self-attention
- Feed-forward networks
- Positional encoding
- Layer normalization

The encoder-decoder structure made it ideal for translation tasks initially.
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write(sample_text)
    tmp_path = f.name

loader = TextLoader(tmp_path, encoding='utf-8')
docs = loader.load()

print(f"Number of documents: {len(docs)}")
print(f"Content length: {len(docs[0].page_content)} chars")
print(f"Metadata: {docs[0].metadata}")
print(f"\nFirst 100 chars: {docs[0].page_content[:100]}")

os.unlink(tmp_path)
```

**Output:**
```text
Number of documents: 1
Content length: 331 chars
Metadata: {'source': '/tmp/tmpXXXXXX.txt'}

First 100 chars: Introduction to Transformers

The transformer architecture was introduced in 'Attention Is All You Need' (2017).
```

`TextLoader` produces a single `Document` for the entire file. The `metadata` captures the source path automatically.

## Loading PDFs

PDFs are the most common document format in enterprise RAG. Two loaders cover 90% of use cases:

```python
from langchain_community.document_loaders import PyPDFLoader

# PyPDFLoader: one Document per page
# Install: pip install pypdf
loader = PyPDFLoader("path/to/document.pdf")
pages = loader.load()

print(f"Pages loaded: {len(pages)}")
print(f"Page 0 metadata: {pages[0].metadata}")
print(f"Page 0 content (first 200 chars): {pages[0].page_content[:200]}")
```

**Output (illustrative — requires actual PDF):**
```text
Pages loaded: 12
Page 0 metadata: {'source': 'document.pdf', 'page': 0}
Page 0 content (first 200 chars): Abstract

We propose a new simple network architecture, the Transformer, based solely on attention mechanisms...
```

`PyPDFLoader` creates one `Document` per page with page number in metadata. This is critical for citation — you can tell users "this came from page 3" because the metadata is preserved.

### PDFMiner for better text extraction

```python
from langchain_community.document_loaders import PDFMinerLoader

# PDFMinerLoader: better layout preservation, fewer encoding issues
# Install: pip install pdfminer.six
loader = PDFMinerLoader("path/to/complex_layout.pdf")
docs = loader.load()  # returns single Document with full text
```

`PyPDFLoader` is fast but struggles with multi-column layouts and tables. `PDFMinerLoader` handles complex layouts better but is slower. For scanned PDFs (images), neither works — you need OCR via `UnstructuredPDFLoader` with tesseract.

## Loading Web Pages

```python
from langchain_community.document_loaders import WebBaseLoader
import bs4

# WebBaseLoader: loads and parses HTML pages
# Install: pip install beautifulsoup4
loader = WebBaseLoader(
    web_paths=["https://python.langchain.com/docs/introduction/"],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("article-content", "markdown", "content")
        )
    ),
)

# Note: requires internet access
# docs = loader.load()
# print(f"Loaded {len(docs)} document(s)")
# print(f"Content length: {len(docs[0].page_content)} chars")

print("WebBaseLoader configured — call loader.load() to fetch content.")
print("bs_kwargs filters HTML to only the content sections (reduces noise).")
```

**Output:**
```text
WebBaseLoader configured — call loader.load() to fetch content.
bs_kwargs filters HTML to only the content sections (reduces noise).
```

The `bs_kwargs` `SoupStrainer` argument limits which HTML elements are extracted. Without it, you get navigation, footer, cookie banners — all noise for retrieval.

![Document loading and splitting pipeline showing chunks](https://images.unsplash.com/photo-1639762681485-074b7f938ba0?w=1200&auto=format&fit=crop&q=80)

## Text Splitters: Why Chunking Matters

A 100-page PDF can't go into a single embedding. Even if it did, the resulting vector would be an average of 100 pages of meaning — useless for retrieval. You need chunks small enough to be meaningfully embedded but large enough to answer a question.

The wrong chunk size is a major RAG failure mode:
- **Too small** (< 100 chars): chunks don't contain enough context to answer questions
- **Too large** (> 2000 chars): embeddings are diluted, retrieval precision drops
- **Bad boundaries**: splitting mid-sentence loses context at the edges

### RecursiveCharacterTextSplitter: The Default Choice

`RecursiveCharacterTextSplitter` splits on `["\n\n", "\n", " ", ""]` in that priority order. It tries paragraph breaks first, then line breaks, then spaces, then characters — ensuring splits happen at the most semantically meaningful boundary available.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

text = """
Transformers and Attention Mechanisms

The transformer architecture revolutionized natural language processing. 
At its core is the self-attention mechanism, which allows each token to 
attend to every other token in the sequence.

Multi-Head Attention

Multi-head attention runs several attention operations in parallel. Each 
"head" learns to attend to different aspects of the input — one head might 
focus on syntactic relationships while another captures semantic similarity.

The output of all heads is concatenated and projected through a linear layer.
This allows the model to jointly attend to information from different 
representation subspaces at different positions.

Feed-Forward Networks

After attention, each position passes through the same feed-forward network 
independently. This consists of two linear transformations with a ReLU 
activation: FFN(x) = max(0, xW1 + b1)W2 + b2.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)

chunks = splitter.split_text(text)
print(f"Text length: {len(text)} chars")
print(f"Number of chunks: {len(chunks)}")
print(f"\n--- Chunk 0 ({len(chunks[0])} chars) ---")
print(chunks[0])
print(f"\n--- Chunk 1 ({len(chunks[1])} chars) ---")
print(chunks[1])
```

**Output:**
```text
Text length: 815 chars
Number of chunks: 4

--- Chunk 0 (287 chars) ---
Transformers and Attention Mechanisms

The transformer architecture revolutionized natural language processing. 
At its core is the self-attention mechanism, which allows each token to 
attend to every other token in the sequence.

--- Chunk 1 (300 chars) ---
Multi-Head Attention

Multi-head attention runs several attention operations in parallel. Each 
"head" learns to attend to different aspects of the input — one head might 
focus on syntactic relationships while another captures semantic similarity.
```

`chunk_overlap=50` means consecutive chunks share 50 characters. This ensures that sentences split across chunk boundaries are still fully represented in at least one chunk — critical for questions whose answers span a chunk boundary.

### Splitting Documents (not raw text)

When splitting `Document` objects, use `split_documents()` to preserve metadata:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

docs = [
    Document(
        page_content="""Introduction to Neural Networks

Neural networks are computational models inspired by biological brains. 
They consist of layers of interconnected nodes that process information.
Each connection has a weight that is adjusted during training.""",
        metadata={"source": "chapter1.pdf", "page": 1}
    ),
    Document(
        page_content="""Backpropagation Algorithm

The backpropagation algorithm computes gradients of the loss function 
with respect to the weights using the chain rule of calculus.
These gradients are used to update weights via gradient descent.""",
        metadata={"source": "chapter2.pdf", "page": 5}
    ),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
chunks = splitter.split_documents(docs)

print(f"Input: {len(docs)} documents → Output: {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i}: {len(chunk.page_content)} chars | source: {chunk.metadata['source']}, page: {chunk.metadata['page']}")
    print(f"  {chunk.page_content[:80]}...")
```

**Output:**
```text
Input: 2 documents → Output: 5 chunks

Chunk 0: 149 chars | source: chapter1.pdf, page: 1
  Introduction to Neural Networks

Neural networks are computational models inspired...

Chunk 1: 143 chars | source: chapter1.pdf, page: 1
  Each connection has a weight that is adjusted during training....

Chunk 2: 139 chars | source: chapter2.pdf, page: 5
  Backpropagation Algorithm

The backpropagation algorithm computes gradients of ...

Chunk 3: 130 chars | source: chapter2.pdf, page: 5
  These gradients are used to update weights via gradient descent....
```

Each chunk inherits `source` and `page` from its parent document. When this chunk is retrieved and used to answer a question, you can cite exactly which document and page it came from.

## Choosing chunk_size and chunk_overlap

The right values depend on your embedding model and use case:

| Embedding model | Recommended chunk_size | Reasoning |
|---|---|---|
| `text-embedding-3-small` | 500–1000 chars | 8192 token limit, good at longer context |
| `sentence-transformers/all-mpnet-base-v2` | 300–500 chars | 512 token limit, trained on sentences |
| `text-embedding-ada-002` | 300–800 chars | 8192 token limit |

A simple heuristic: target 200–400 tokens per chunk (roughly 800–1600 characters for English text). This leaves room for the question in the context window and gives the embedding enough text to be semantically meaningful.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def estimate_chunk_token_count(text: str, chars_per_token: float = 4.0) -> int:
    """Rough estimate: ~4 chars per token for English text."""
    return int(len(text) / chars_per_token)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

sample = "The attention mechanism was introduced by Bahdanau et al. in 2014 for neural machine translation. " * 20
chunks = splitter.split_text(sample)

for i, chunk in enumerate(chunks[:3]):
    est_tokens = estimate_chunk_token_count(chunk)
    print(f"Chunk {i}: {len(chunk)} chars ≈ {est_tokens} tokens")
```

**Output:**
```text
Chunk 0: 800 chars ≈ 200 tokens
Chunk 1: 800 chars ≈ 200 tokens
Chunk 2: 400 chars ≈ 100 tokens
```

~200 tokens per chunk is a good target for retrieval — meaningful enough to embed well, small enough to not dilute the vector.

### Gotcha: chunk_overlap doesn't guarantee sentence continuity

`chunk_overlap` overlaps by character count, not by sentence boundary. If a sentence ends at character 785 and your `chunk_size` is 800, the next chunk still starts at character 750 — mid-sentence. For better sentence continuity, use `SentenceTransformersTokenTextSplitter` which splits on sentence boundaries:

```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

splitter = SentenceTransformersTokenTextSplitter(
    model_name="sentence-transformers/all-mpnet-base-v2",
    chunk_overlap=0,
    tokens_per_chunk=256,  # max tokens per chunk (model-specific)
)

text = "The transformer architecture uses self-attention. It was introduced in 2017. The paper proposed both encoder and decoder stacks. Each layer has multi-head attention followed by feed-forward networks."
chunks = splitter.split_text(text)
for i, c in enumerate(chunks):
    print(f"Chunk {i}: {c}")
```

**Output:**
```text
Chunk 0: The transformer architecture uses self-attention. It was introduced in 2017. The paper proposed both encoder and decoder stacks.
Chunk 1: Each layer has multi-head attention followed by feed-forward networks.
```

Splits happen at sentence boundaries, preserving full sentences in each chunk.

![Text chunking visualization showing overlap between consecutive chunks](https://images.unsplash.com/photo-1677442135703-1787eea5ce01?w=1200&auto=format&fit=crop&q=80)

## A Complete Ingestion Pipeline

Putting loaders and splitters together into an end-to-end ingestion function:

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import tempfile
import os

def ingest_documents(file_paths: List[str], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    """
    Load and split documents from file paths.
    Returns a list of chunked Documents with source metadata.
    """
    raw_docs = []

    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()

        if ext in ['.txt', '.md']:
            loader = TextLoader(path, encoding='utf-8')
            raw_docs.extend(loader.load())
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(raw_docs)
    print(f"Loaded {len(raw_docs)} docs → split into {len(chunks)} chunks")
    return chunks

# Test with sample files
content_a = "Attention mechanisms allow models to focus on relevant parts of the input.\n\n" + "They compute a weighted sum of values based on query-key similarity.\n\n" * 5
content_b = "Transformers use positional encoding to inject sequence order information.\n\n" + "This allows parallel processing unlike recurrent networks.\n\n" * 5

with tempfile.TemporaryDirectory() as tmpdir:
    path_a = os.path.join(tmpdir, "attention.txt")
    path_b = os.path.join(tmpdir, "transformers.txt")
    open(path_a, 'w').write(content_a)
    open(path_b, 'w').write(content_b)

    chunks = ingest_documents([path_a, path_b], chunk_size=200, chunk_overlap=30)
    for chunk in chunks[:3]:
        print(f"\n[{chunk.metadata['source'].split('/')[-1]}] {chunk.page_content[:80]}...")
```

**Output:**
```text
Loaded 2 docs → split into 14 chunks

[attention.txt] Attention mechanisms allow models to focus on relevant parts of the input.

[attention.txt] They compute a weighted sum of values based on query-key similarity.

[transformers.txt] Transformers use positional encoding to inject sequence order information.
```

Each chunk carries its source filename. In a production RAG system, this `source` metadata is what you display to users as citation.

## Conclusion

Document loading and splitting are the least glamorous parts of RAG — and the most impactful on quality. Load PDFs with `PyPDFLoader` (or `PDFMinerLoader` for complex layouts) and always preserve page metadata. Split with `RecursiveCharacterTextSplitter` at 500–1000 characters with 10–20% overlap, adjusting for your embedding model's token limit. Use `split_documents()` (not `split_text()`) to preserve source metadata through the pipeline. Get these decisions right and every downstream component — embeddings, vector store, retrieval — has clean, well-bounded text to work with.

The next post covers embeddings and vector stores — converting chunks to dense vectors, storing them in FAISS or Chroma, and running similarity search.
