---
title: >-
  Working with LLMs and Chat Models in LangChain: OpenAI, Anthropic, and Local
  Models via Ollama
excerpt: >-
  LangChain wraps every LLM provider behind the same Runnable interface. Swap
  OpenAI for Claude or a local Llama model without changing a line of your chain
  logic.
author: Soham Sharma
authorName: Soham Sharma
category: AI
tags:
  - LangChain
  - OpenAI
  - Anthropic
  - Ollama
  - LLM
  - Local Models
status: published
featuredImage: >-
  https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=1200&auto=format&fit=crop&q=80
colab_notebook: >-
  https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/langchain/langchain_3.ipynb
series_id: langchain-production
series_slug: langchain-production
series_title: LangChain / LangSmith / LangGraph — In Production
difficulty: beginner
week: null
day: 12
tools:
  - LangChain
  - OpenAI
  - Anthropic
  - Ollama
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/langchain/langchain_3.ipynb)


The most underrated feature of LangChain is model portability. You build a chain against the `ChatOpenAI` interface, and switching to Claude or a locally-hosted Llama model is a single-line change. This works because every LLM in LangChain implements the same `Runnable` protocol with identical `.invoke()`, `.stream()`, and `.batch()` methods. Understanding how each provider is wired up — including their differences in parameter names and token counting — is what makes that portability practical rather than aspirational.

## The BaseChatModel Interface

All chat model classes in LangChain extend `BaseChatModel`. The key methods:

| Method | Returns | Use case |
|---|---|---|
| `.invoke(messages)` | `AIMessage` | Single synchronous call |
| `.stream(messages)` | Generator of `AIMessageChunk` | Token-by-token streaming |
| `.batch([msgs1, msgs2])` | List of `AIMessage` | Parallel calls, rate-limited |
| `.ainvoke(messages)` | `AIMessage` (async) | Async frameworks |

Every model class also exposes a `.bind()` method that attaches fixed parameters (tools, stop sequences, temperature) to the model without calling it.

## OpenAI: ChatOpenAI

`ChatOpenAI` wraps the OpenAI chat completions API. Install: `pip install langchain-openai`.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=256,
    # api_key="sk-..." # or set OPENAI_API_KEY env var
)

messages = [
    SystemMessage(content="Be concise. Answer in 2 sentences max."),
    HumanMessage(content="What is gradient descent?"),
]

response = model.invoke(messages)
print(type(response))
print(response.content)
print(f"\nToken usage: {response.response_metadata.get('token_usage', {})}")
```

**Output:**
```text
<class 'langchain_core.messages.ai.AIMessage'>
Gradient descent is an iterative optimization algorithm that minimizes a function by repeatedly moving in the direction of the negative gradient. In machine learning, it updates model parameters to reduce the loss function by small steps proportional to the learning rate.

Token usage: {'completion_tokens': 47, 'prompt_tokens': 31, 'total_tokens': 78}
```

> Note: LLM output content varies by run.

`response_metadata` contains the token usage, model name, and finish reason. This is where you track costs in production.

### ChatOpenAI configuration options

```python
from langchain_openai import ChatOpenAI

# With logprobs and streaming
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=512,
    logprobs=True,       # returns token log probabilities
    top_p=0.95,
    frequency_penalty=0.1,
    presence_penalty=0.1,
    timeout=30,          # request timeout in seconds
    max_retries=3,       # automatic retry on rate limits
)

print(f"Model: {model.model_name}")
print(f"Temperature: {model.temperature}")
```

**Output:**
```text
Model: gpt-4o
Temperature: 0.7
```

## Anthropic: ChatAnthropic

`ChatAnthropic` wraps the Anthropic Claude API. Install: `pip install langchain-anthropic`.

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatAnthropic(
    model="claude-3-5-haiku-20241022",
    temperature=0,
    max_tokens=256,
    # api_key="sk-ant-..." # or set ANTHROPIC_API_KEY env var
)

messages = [
    SystemMessage(content="Be concise. Answer in 2 sentences max."),
    HumanMessage(content="What is attention mechanism in transformers?"),
]

response = model.invoke(messages)
print(response.content)
print(f"\nUsage: {response.usage_metadata}")
```

**Output:**
```text
The attention mechanism allows each token in a sequence to weigh the relevance of all other tokens when computing its representation, enabling the model to capture long-range dependencies. It computes query-key-value dot products to produce weighted sums of value vectors, with softmax normalization ensuring the weights sum to one.

Usage: {'input_tokens': 28, 'output_tokens': 59, 'total_tokens': 87}
```

> Note: LLM output content varies by run.

The interface is identical to `ChatOpenAI` — same `invoke()`, same `AIMessage` return type, same usage metadata pattern. The difference is in the constructor parameters (model names, `max_tokens` vs no max, Claude-specific features like extended thinking).

### Swapping models in a chain: zero chain code change

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Explain {concept} in one sentence.")
parser = StrOutputParser()

# Swap models here — chain code is identical
openai_chain  = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)  | parser
claude_chain  = prompt | ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0) | parser

input_data = {"concept": "backpropagation"}

openai_result = openai_chain.invoke(input_data)
claude_result = claude_chain.invoke(input_data)

print(f"OpenAI:  {openai_result}")
print(f"Claude:  {claude_result}")
```

**Output:**
```text
OpenAI:  Backpropagation is the algorithm that computes gradients of the loss with respect to each parameter by applying the chain rule backwards through the network.
Claude:  Backpropagation is an algorithm that efficiently calculates how much each parameter in a neural network contributed to the error, enabling gradient-based optimization.
```

> Note: LLM outputs vary by run.

Same chain template, same parser, same input — only the model line changed. This is the portability LangChain was designed for.

![LLM provider comparison showing OpenAI, Anthropic, and local models](https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=1200&auto=format&fit=crop&q=80)

## Local Models via Ollama

Ollama runs open-source models (Llama, Mistral, Phi, Gemma) locally with a REST API. No cloud, no API key, no cost per token. Install Ollama, pull a model, and LangChain talks to it via `ChatOllama`.

Install: `pip install langchain-ollama` and [Ollama](https://ollama.com) on your machine.

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# Requires Ollama running locally (ollama serve)
# Pull model first: ollama pull llama3.2
model = ChatOllama(
    model="llama3.2",
    temperature=0,
    base_url="http://localhost:11434",  # default Ollama endpoint
)

response = model.invoke([HumanMessage(content="What is 12 * 15?")])
print(response.content)
```

**Output:**
```text
12 * 15 = 180
```

> Note: Requires Ollama running locally with llama3.2 pulled. Output may vary.

`ChatOllama` implements the same `BaseChatModel` interface — your existing chains work with zero modification.

### Running all three in a benchmark loop

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

prompt = ChatPromptTemplate.from_template(
    "Answer in exactly one sentence: {question}"
)
parser = StrOutputParser()

models = {
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0),
    "claude-haiku": ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0),
    # "llama3.2": ChatOllama(model="llama3.2", temperature=0),  # uncomment if Ollama installed
}

question = {"question": "Why do neural networks use non-linear activation functions?"}

for name, model in models.items():
    chain = prompt | model | parser
    start = time.time()
    result = chain.invoke(question)
    latency = time.time() - start
    print(f"\n[{name}] ({latency:.2f}s)")
    print(f"  {result}")
```

**Output:**
```text
[gpt-4o-mini] (0.87s)
  Neural networks use non-linear activation functions because linear transformations can only represent linear functions regardless of depth, while non-linearity enables them to approximate arbitrary complex functions.

[claude-haiku] (1.12s)
  Non-linear activation functions allow neural networks to learn complex, non-linear patterns and representations that purely linear transformations cannot capture.
```

> Note: Latencies and content vary by run and network conditions.

## bind(): Attaching Parameters to a Model

`.bind()` creates a new Runnable with parameters pre-attached. Use it to configure stop sequences, tools, or format options without modifying the chain structure.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Bind stop sequences — model stops when it generates "---"
model_with_stop = model.bind(stop=["---"])

chain = (
    ChatPromptTemplate.from_template("List 5 Python libraries for {task}. Separate with '---'.")
    | model_with_stop
    | StrOutputParser()
)

result = chain.invoke({"task": "data science"})
print(result)
print(f"Stopped at separator: {'---' not in result}")
```

**Output:**
```text
NumPy
Stopped at separator: True
```

The model generated "NumPy" and then started generating "---", which triggered the stop sequence. Only the content before the stop sequence is returned.

## Gotchas: Provider Differences

### 1. System message handling

Claude treats `SystemMessage` differently than OpenAI — it maps to Anthropic's `system` parameter rather than a message in the conversation. Both work via LangChain's abstraction, but raw API calls differ.

### 2. token counting

Different providers count tokens differently. OpenAI uses tiktoken; Anthropic uses their own tokenizer. For accurate cost tracking, use the provider's native token counter rather than estimating from character count.

### 3. Streaming chunks for OpenAI vs Anthropic

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("Count from 1 to 5, one number per line.")
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)

chunks = []
for chunk in chain.stream({}):
    chunks.append(chunk)

print(f"Number of chunks received: {len(chunks)}")
print(f"First 5 chunks: {chunks[:5]}")
print(f"Assembled: {''.join(chunks)}")
```

**Output:**
```text
Number of chunks received: 14
First 5 chunks: ['', '1', '\n', '2', '\n']
Assembled: 1
2
3
4
5
```

> Note: Chunk count varies by model and network conditions.

OpenAI streams at approximately the token boundary. Anthropic streams at similar granularity. Ollama streaming cadence depends on the model and hardware.

### 4. Context window limits

| Model | Context window |
|---|---|
| gpt-4o | 128K tokens |
| claude-3-5-sonnet | 200K tokens |
| llama3.2 (3B) | 128K tokens |
| mistral-7b | 32K tokens |

Exceeding the context window raises a provider-specific error. LangChain does not automatically truncate — you need to manage context length explicitly in your chain.

## Conclusion

Every LLM in LangChain speaks the same `Runnable` interface. Switching from OpenAI to Claude to a local Llama model is a constructor swap, not a refactor. The practical differences — model names, token counting, context limits, streaming granularity — are worth knowing but don't change the chain architecture. `bind()` attaches provider-specific parameters cleanly without polluting chain logic. Build your chains against the interface, not the provider, and model selection becomes a deployment decision rather than an architectural one.

The next post covers document loaders and text splitters — loading PDFs, web pages, and raw text, then splitting them into chunks that fit in context windows.
