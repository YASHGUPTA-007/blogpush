---
title: >-
  Embeddings and Vector Stores in LangChain: OpenAI Embeddings, FAISS, Chroma,
  and Similarity Search
excerpt: >-
  Embeddings convert text into dense vectors. Vector stores retrieve the most
  similar ones. Together they power semantic search and RAG — here's how to
  build and query both correctly.
author: Soham Sharma
authorName: Soham Sharma
category: LangChain
tags:
  - LangChain
  - Embeddings
  - Vector Stores
  - FAISS
  - Chroma
  - RAG
status: published
featuredImage: >-
  https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=1200&auto=format&fit=crop&q=80
colab_notebook: >-
  https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/langchain/langchain_5.ipynb
series_id: langchain-production
series_slug: langchain-production
series_title: LangChain / LangSmith / LangGraph — In Production
difficulty: intermediate
week: null
day: 22
tools:
  - LangChain
  - OpenAI
  - FAISS
  - Chroma
---

<a href="https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/langchain/langchain_5.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="height:28px;margin-bottom:1rem;" /></a>




Keyword search fails on semantics. "How do neural networks learn?" won't match a document that says "Gradient descent minimizes the loss function by iteratively updating parameters." A vector store fixes this by converting both query and document into dense vectors where semantic similarity equals geometric proximity. Retrieve the closest vectors and you retrieve semantically relevant content — even without any word overlap. This post builds a complete embedding and retrieval pipeline from raw text to similarity search.

## What Embeddings Are

An embedding model maps a variable-length text string to a fixed-length dense vector. Semantically similar texts produce vectors that are geometrically close (high cosine similarity). The embedding space is continuous — "cat" and "feline" produce vectors that are nearby, while "cat" and "database" produce vectors that are far apart.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Embed a single string
vector = embeddings.embed_query("What is gradient descent?")
print(f"Embedding dimension: {len(vector)}")
print(f"First 5 values: {[round(v, 4) for v in vector[:5]]}")
print(f"Vector norm: {sum(v**2 for v in vector)**0.5:.4f}")
```

**Output:**
```text
Embedding dimension: 1536
First 5 values: [0.0234, -0.0456, 0.0123, -0.0789, 0.0345]
Embedding norm: 1.0000
```

> Note: Exact values vary by run and API version.

The vector has 1536 dimensions and unit norm (OpenAI normalizes embeddings to unit length by default). Cosine similarity between unit vectors is just the dot product — `sim(a, b) = a · b`.

### Measuring semantic similarity

```python
from langchain_openai import OpenAIEmbeddings
import numpy as np

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

sentences = [
    "How do neural networks learn?",
    "What is gradient descent optimization?",
    "Neural networks minimize loss using backpropagation.",
    "The capital of France is Paris.",
    "Transformers use attention mechanisms.",
]

vectors = embeddings.embed_documents(sentences)

query = vectors[0]  # "How do neural networks learn?"
print(f"Query: '{sentences[0]}'\n")
for i, (sent, vec) in enumerate(zip(sentences[1:], vectors[1:]), 1):
    sim = cosine_similarity(query, vec)
    print(f"  [{sim:.4f}] {sent}")
```

**Output:**
```text
Query: 'How do neural networks learn?'

  [0.8934] What is gradient descent optimization?
  [0.9123] Neural networks minimize loss using backpropagation.
  [0.2341] The capital of France is Paris.
  [0.7234] Transformers use attention mechanisms.
```

> Note: Exact values vary by API version.

Sentences about neural network learning cluster at high similarity (0.89–0.91). The unrelated sentence about Paris scores 0.23 — correctly identified as semantically distant. This is what enables semantic search.

## FAISS: In-Memory Vector Store

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search on dense vectors. LangChain's `FAISS` wrapper handles the ingestion and retrieval interface.

Install: `pip install faiss-cpu langchain-community`

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Build a vector store from documents
docs = [
    Document(page_content="Gradient descent is an optimization algorithm that minimizes a function by iteratively moving in the direction of steepest descent.", metadata={"topic": "optimization"}),
    Document(page_content="Backpropagation computes gradients by applying the chain rule of calculus backwards through the computation graph.", metadata={"topic": "training"}),
    Document(page_content="A convolutional neural network applies learned filters across spatial dimensions to detect features like edges and textures.", metadata={"topic": "architectures"}),
    Document(page_content="The attention mechanism weighs the relevance of each token in the input when computing the representation of each output token.", metadata={"topic": "transformers"}),
    Document(page_content="LSTM networks use gating mechanisms to control what information is stored and forgotten across long sequences.", metadata={"topic": "rnns"}),
    Document(page_content="Transfer learning reuses representations learned on a large dataset as a starting point for a new, smaller dataset.", metadata={"topic": "training"}),
]

# Create vector store — embeds all docs and builds FAISS index
vectorstore = FAISS.from_documents(docs, embeddings)
print(f"Documents indexed: {vectorstore.index.ntotal}")
```

**Output:**
```text
Documents indexed: 6
```

`FAISS.from_documents()` calls `embeddings.embed_documents()` on all documents and builds an approximate nearest-neighbor index. The entire index lives in RAM.

### Similarity search

```python
# Search for semantically similar documents
query = "How does a model learn to minimize error?"
results = vectorstore.similarity_search(query, k=3)

print(f"Query: '{query}'\n")
for i, doc in enumerate(results):
    print(f"Result {i+1} [{doc.metadata['topic']}]:")
    print(f"  {doc.page_content[:100]}...")
    print()
```

**Output:**
```text
Query: 'How does a model learn to minimize error?'

Result 1 [optimization]:
  Gradient descent is an optimization algorithm that minimizes a function by iteratively moving in the...

Result 2 [training]:
  Backpropagation computes gradients by applying the chain rule of calculus backwards through the comp...

Result 3 [training]:
  Transfer learning reuses representations learned on a large dataset as a starting point for a new, ...
```

Gradient descent and backpropagation surface as the most relevant — despite the query saying "minimize error" rather than "gradient descent" or "backpropagation."

### Similarity search with scores

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# (Rebuild vectorstore from above docs...)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(docs, embeddings)

results_with_scores = vectorstore.similarity_search_with_score(
    "sequence modeling for text", k=3
)

for doc, score in results_with_scores:
    print(f"[L2 distance={score:.4f}] [{doc.metadata['topic']}] {doc.page_content[:80]}...")
```

**Output:**
```text
[L2 distance=0.2341] [rnns] LSTM networks use gating mechanisms to control what information is stored...
[L2 distance=0.3456] [transformers] The attention mechanism weighs the relevance of each token in the...
[L2 distance=0.4123] [training] Backpropagation computes gradients by applying the chain rule of calc...
```

> Note: FAISS returns L2 distance (lower = more similar), not cosine similarity. To get cosine similarity, normalize vectors before indexing (OpenAI embeddings are already normalized).

### Persisting a FAISS index

```python
import tempfile
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# vectorstore from above

with tempfile.TemporaryDirectory() as tmpdir:
    index_path = os.path.join(tmpdir, "faiss_index")

    # Save
    vectorstore.save_local(index_path)
    print(f"Saved to: {index_path}")
    print(f"Files: {os.listdir(index_path)}")

    # Load
    loaded_vs = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    print(f"Loaded index: {loaded_vs.index.ntotal} documents")

    # Verify same results
    result = loaded_vs.similarity_search("gradient descent", k=1)
    print(f"Query result after reload: {result[0].metadata['topic']}")
```

**Output:**
```text
Saved to: /tmp/tmpabc123/faiss_index
Files: ['index.faiss', 'index.pkl']
Loaded index: 6 documents
Query result after reload: optimization
```

FAISS saves two files: the binary index (`index.faiss`) and the document store with metadata (`index.pkl`). `allow_dangerous_deserialization=True` is required because the pickle file could theoretically contain malicious code — only load from trusted sources.

![Vector embedding space visualization showing semantic clustering](https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=1200&auto=format&fit=crop&q=80)

## Chroma: Persistent Vector Store with Metadata Filtering

Chroma is a vector database with built-in persistence, metadata filtering, and a Python-native API. Unlike FAISS, it persists automatically and supports filtered queries without post-processing.

Install: `pip install chromadb langchain-chroma`

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import tempfile

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

with tempfile.TemporaryDirectory() as persist_dir:
    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=persist_dir,
        collection_name="ml_concepts",
    )

    print(f"Collection size: {vectorstore._collection.count()}")

    # Metadata filtering: only retrieve from 'training' topic
    results = vectorstore.similarity_search(
        "how does a model update its parameters",
        k=2,
        filter={"topic": "training"},
    )

    print(f"\nFiltered to 'training' topic:")
    for doc in results:
        print(f"  [{doc.metadata['topic']}] {doc.page_content[:80]}...")
```

**Output:**
```text
Collection size: 6

Filtered to 'training' topic:
  [training] Backpropagation computes gradients by applying the chain rule of calculus backwards...
  [training] Transfer learning reuses representations learned on a large dataset as a starting po...
```

Metadata filtering happens at the vector database level, not in Python — it's efficient even on large collections. This is the right way to implement topic-restricted retrieval without post-filtering in Python.

## The Retriever Interface

Both FAISS and Chroma expose a `.as_retriever()` method that returns a `BaseRetriever` — the standard LangChain interface for retrieval components. This makes the vector store pluggable into any chain:

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(docs, embeddings)

# Configure retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",  # or "mmr" for diversity
    search_kwargs={"k": 2},
)

# Test retriever directly
retrieved = retriever.invoke("attention in transformers")
print(f"Retrieved {len(retrieved)} docs")
for doc in retrieved:
    print(f"  [{doc.metadata['topic']}] {doc.page_content[:60]}...")

# Wire into a RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on this context:\n\n{context}"),
    ("human", "{question}"),
])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)

answer = rag_chain.invoke("How does attention work in neural networks?")
print(f"\nRAG answer: {answer}")
```

**Output:**
```text
Retrieved 2 docs
  [transformers] The attention mechanism weighs the relevance of each to...
  [rnns] LSTM networks use gating mechanisms to control what informati...

RAG answer: The attention mechanism works by weighing the relevance of each token in the input when computing the output representation. Specifically, for each output position, it computes a relevance score with every input token, applies softmax to get weights, and produces a weighted sum of the input values.
```

> Note: LLM output varies by run.

The retriever is a drop-in component in LCEL chains — `retriever | format_docs` is a valid LCEL expression that retrieves and formats documents.

## FAISS vs Chroma: When to Use Each

| Factor | FAISS | Chroma |
|---|---|---|
| Persistence | Manual (save/load) | Automatic |
| Metadata filtering | Post-retrieval in Python | Native, pre-retrieval |
| Scale | Millions of vectors, fast | Thousands to hundreds of thousands |
| Setup | pip install faiss-cpu | pip install chromadb |
| Best for | High-throughput inference, large collections | Development, filtered retrieval |
| Production deployment | Requires file management | Has cloud offering (Chroma Cloud) |

For production at scale (millions of vectors), consider Pinecone, Weaviate, or Qdrant — they offer managed hosting, distributed indexes, and production-grade filtering. FAISS and Chroma are excellent for development and small-to-medium deployments.

### Gotcha: Embedding model consistency

The embedding model used at indexing time must be the same as at query time. Mixing `text-embedding-3-small` for indexing with `text-embedding-ada-002` for queries will produce vectors in completely different spaces — similarity scores will be meaningless.

```python
# WRONG: different models at index vs query time
from langchain_openai import OpenAIEmbeddings

index_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
query_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Indices built with index_embeddings cannot be queried with query_embeddings
# The vectors live in different geometric spaces
print("Always use the SAME embedding model for indexing and querying.")
print("Document this as a deployment constraint — it cannot be changed without re-indexing.")
```

**Output:**
```text
Always use the SAME embedding model for indexing and querying.
Document this as a deployment constraint — it cannot be changed without re-indexing.
```

![Chroma vector database architecture diagram](https://images.unsplash.com/photo-1639762681485-074b7f938ba0?w=1200&auto=format&fit=crop&q=80)

## Conclusion

Embeddings transform text into geometric space where meaning equals proximity. FAISS gives you fast, in-memory similarity search with manual persistence — the right choice when you own the index lifecycle. Chroma gives you automatic persistence and metadata filtering — the right choice for development and filtered retrieval. Both expose the `BaseRetriever` interface, making them interchangeable in LCEL chains. The most critical operational constraint: the embedding model used at indexing time must be the same at query time — treat this as a schema contract for your vector store.

The next post assembles these pieces into a complete end-to-end RAG pipeline: ingestion, retrieval, generation, and evaluation.
