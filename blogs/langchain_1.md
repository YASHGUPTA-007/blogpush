---
title: >-
  LangChain Architecture Overview: Chains, Runnables, LCEL, and the New vs Old
  API
excerpt: >-
  LangChain's architecture changed fundamentally with v0.1. Learn chains,
  runnables, and LCEL so you build on the current API — not the deprecated one.
author: Soham Sharma
authorName: Soham Sharma
category: AI
tags:
  - LangChain
  - LLM
  - Python
  - LCEL
  - Runnables
status: published
featuredImage: >-
  https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=1200&auto=format&fit=crop&q=80
colab_notebook: >-
  https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/langchain/langchain_1.ipynb
series_id: langchain-production
series_slug: langchain-production
series_title: LangChain / LangSmith / LangGraph — In Production
difficulty: beginner
week: null
day: 2
tools:
  - LangChain
  - OpenAI
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/langchain/langchain_1.ipynb)


Most LangChain tutorials on the internet are outdated. They teach `LLMChain`, `SequentialChain`, and `ConversationChain` — classes that still exist but are officially deprecated in favor of a completely different paradigm called **LangChain Expression Language (LCEL)**. If you're building something that will run in production next year, you need to understand the current architecture. This post gives you that foundation.

## The Architecture Shift: Why LCEL Replaced Chains

The old LangChain (pre-0.1) used Python class inheritance to compose operations. You'd subclass `Chain`, override `_call`, and wire together `LLMChain`, `TransformChain`, and `SequentialChain`. This worked but had serious problems:

- **No native streaming** — you had to bolt it on per-chain
- **Hard to introspect** — intermediate outputs were buried in `chain.run()` return values
- **Difficult to parallelize** — no built-in map/reduce
- **LangSmith tracing was fragile** — not all chains produced clean run trees

LCEL solves all of these with a single abstraction: the **Runnable interface**.

## The Runnable Interface

Every component in modern LangChain implements the `Runnable` protocol. A Runnable has three core methods:

| Method | What it does |
|---|---|
| `.invoke(input)` | Single synchronous call |
| `.stream(input)` | Returns a generator of output chunks |
| `.batch(inputs)` | Processes a list of inputs, potentially in parallel |

Async variants (`.ainvoke`, `.astream`, `.abatch`) are available for every Runnable.

Because everything is a Runnable — prompts, models, parsers, retrievers — they all compose the same way. That's the core insight.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Each of these is a Runnable
prompt = ChatPromptTemplate.from_template("Explain {concept} in one sentence.")
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

# The pipe operator | composes Runnables left-to-right
chain = prompt | model | parser

result = chain.invoke({"concept": "attention mechanism"})
print(result)
```

**Output:**
```text
The attention mechanism allows a model to weigh the importance of different input tokens when generating each output token, enabling it to focus on relevant parts of the input sequence.
```

The `|` operator is syntactic sugar for `RunnableSequence`. `prompt | model | parser` creates a pipeline where the output of each step becomes the input of the next.

![LangChain LCEL pipeline diagram](https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=1200&auto=format&fit=crop&q=80)

## Understanding LCEL: LangChain Expression Language

LCEL is not a separate language — it's a set of Python operators and classes for composing Runnables. The `|` pipe, `RunnableParallel`, `RunnablePassthrough`, and `RunnableLambda` are the building blocks.

### RunnablePassthrough: Passing input unchanged

`RunnablePassthrough` forwards its input unchanged. It's useful for passing original context through a pipeline while other branches transform it.

```python
from langchain_core.runnables import RunnablePassthrough

passthrough = RunnablePassthrough()
result = passthrough.invoke({"key": "value"})
print(result)
```

**Output:**
```text
{'key': 'value'}
```

### RunnableParallel: Running steps concurrently

`RunnableParallel` takes a dict of Runnables and runs them in parallel, combining their outputs into a single dict.

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

pros_chain = (
    ChatPromptTemplate.from_template("List 2 pros of {technology} in bullet points.")
    | model | parser
)
cons_chain = (
    ChatPromptTemplate.from_template("List 2 cons of {technology} in bullet points.")
    | model | parser
)

parallel = RunnableParallel(
    pros=pros_chain,
    cons=cons_chain,
    original=RunnablePassthrough(),
)

result = parallel.invoke({"technology": "vector databases"})
print("PROS:", result["pros"])
print("CONS:", result["cons"])
print("INPUT:", result["original"])
```

**Output:**
```text
PROS: - High-speed similarity search enables real-time semantic queries
- Efficiently handles high-dimensional embeddings at scale
CONS: - Approximate nearest neighbor search can miss exact matches
- Indexing large datasets requires significant memory overhead
INPUT: {'technology': 'vector databases'}
```

> Note: LLM outputs vary by run.

Both `pros_chain` and `cons_chain` execute concurrently — the total latency is roughly `max(pros_latency, cons_latency)` rather than their sum.

### RunnableLambda: Wrapping arbitrary Python functions

Any Python function can become a Runnable via `RunnableLambda`. This is how you inject custom preprocessing or postprocessing into an LCEL pipeline.

```python
from langchain_core.runnables import RunnableLambda

def word_count(text: str) -> dict:
    return {"text": text, "word_count": len(text.split())}

counter = RunnableLambda(word_count)
print(counter.invoke("The attention mechanism is powerful"))
```

**Output:**
```text
{'text': 'The attention mechanism is powerful', 'word_count': 5}
```

## Prompts: ChatPromptTemplate in Depth

`ChatPromptTemplate` is the standard way to build structured prompts for chat models. It composes a list of messages, each with a role and a template string.

```python
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

system_template = SystemMessagePromptTemplate.from_template(
    "You are an expert in {domain}. Be concise and technical."
)
human_template = HumanMessagePromptTemplate.from_template(
    "Explain: {question}"
)

prompt = ChatPromptTemplate.from_messages([system_template, human_template])

# Format to inspect the messages
messages = prompt.format_messages(domain="distributed systems", question="CAP theorem")
for msg in messages:
    print(f"[{msg.type}] {msg.content}")
```

**Output:**
```text
[system] You are an expert in distributed systems. Be concise and technical.
[human] Explain: CAP theorem
```

The prompt template produces a list of typed messages that chat models expect. The types (`system`, `human`, `ai`) map directly to the roles in the OpenAI and Anthropic APIs.

## Output Parsers

Output parsers transform the raw model response into a structured Python object. The most common:

| Parser | Input | Output |
|---|---|---|
| `StrOutputParser` | `AIMessage` | `str` |
| `JsonOutputParser` | `AIMessage` with JSON | `dict` |
| `PydanticOutputParser` | `AIMessage` with JSON | Pydantic model |
| `CommaSeparatedListOutputParser` | `AIMessage` | `list[str]` |

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

class TechSummary(BaseModel):
    name: str
    year_created: int
    primary_use_case: str

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = JsonOutputParser(pydantic_object=TechSummary)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Return a JSON object matching this schema: {format_instructions}"),
    ("human", "Summarize: {technology}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser

result = chain.invoke({"technology": "PyTorch"})
print(result)
print(type(result))
```

**Output:**
```text
{'name': 'PyTorch', 'year_created': 2016, 'primary_use_case': 'Deep learning research and production model training'}
<class 'dict'>
```

> Note: LLM outputs vary by run.

`parser.get_format_instructions()` generates the JSON schema description automatically from the Pydantic model — you don't write it manually.

## Old API vs New API: What to Avoid

If you're reading older tutorials, you'll see these patterns. Know what they are and use the modern equivalent instead.

| Old API (deprecated) | Modern equivalent |
|---|---|
| `LLMChain(llm=..., prompt=...)` | `prompt \| model \| parser` |
| `chain.run("input")` | `chain.invoke({"key": "input"})` |
| `SequentialChain` | `chain1 \| chain2 \| chain3` |
| `TransformChain` | `RunnableLambda(fn)` |
| `ConversationChain` | `LangGraph` with memory |

The old API still works in LangChain 0.3 but throws deprecation warnings and will be removed. More importantly, the old API does not support streaming, async, or LangSmith tracing as cleanly as LCEL.

### Gotcha: dict vs string inputs

The old `chain.run("string")` accepted a plain string. LCEL chains always accept a dict matching the template variables. This trips up people migrating old code.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("What is {topic}?")
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)

# Correct
result = chain.invoke({"topic": "gradient descent"})

# Wrong — raises ValueError
try:
    chain.invoke("gradient descent")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print(result[:80])
```

**Output (raises):**
```text
Error: KeyError: 'topic'
```

**Output:**
```text
Gradient descent is an optimization algorithm used to minimize a function by iterat
```

## Streaming: A First-Class Feature

One of LCEL's key advantages is that streaming is built into every chain — you don't need a special chain variant.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("Explain {concept} in 3 sentences.")
    | ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    | StrOutputParser()
)

# Stream token by token
for chunk in chain.stream({"concept": "transformer architecture"}):
    print(chunk, end="", flush=True)
print()
```

**Output:**
```text
The transformer architecture is a deep learning model that relies entirely on self-attention mechanisms to process sequential data, eliminating the need for recurrence or convolution. It processes all tokens in a sequence simultaneously, enabling massive parallelization during training. Originally proposed for machine translation in "Attention Is All You Need" (2017), it has since become the foundation for models like BERT, GPT, and T5.
```

> Note: Output streams token by token; shown here as final assembled text.

The same chain you use for `.invoke()` works for `.stream()` without modification. This is the composability LCEL was designed for.

## Inspecting Your Chain

Before running in production, always inspect what your chain will do. LCEL chains expose their input/output schema:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("Summarize {text} in one sentence.")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

print("Input schema:", chain.input_schema.schema())
print("Output schema:", chain.output_schema.schema())
```

**Output:**
```text
Input schema: {'title': 'PromptInput', 'type': 'object', 'properties': {'text': {'title': 'Text', 'type': 'string'}}}
Output schema: {'title': 'StrOutputParserOutput', 'type': 'string'}
```

This is useful for documentation, validation, and LangSmith tracing — the schema tells LangSmith what fields to expect in the run tree.

## Conclusion

LangChain's modern architecture centers on one idea: everything is a Runnable, and Runnables compose with `|`. LCEL gives you streaming, async, batching, and tracing for free on every chain you build, without any special-casing. The old chain classes still work but won't keep pace with the ecosystem — LCEL is where LangGraph, LangSmith, and all new features build on top.

The next post covers prompt templates and output parsers in depth — including Pydantic-based structured output, custom format instructions, and retry logic for malformed responses.
