---
title: "LangChain Architecture Overview: Chains, Runnables, LCEL, and the New API"
excerpt: "Understand LangChain's core architecture — how chains became runnables, what LCEL actually does, and how to navigate the old vs new API split without getting burned."
author: "Soham Sharma"
category: "Technology"
tags: ["LangChain", "LLM", "Python", "AI Agents", "RAG"]
status: "published"
featuredImage: ""
---

LangChain's API has changed dramatically. If you learned it from a tutorial written before mid-2023, much of what you know maps to a deprecated interface. The new model — built around **LangChain Expression Language (LCEL)** and the `Runnable` protocol — is not just syntactic sugar. It's a fundamentally different composition model that enables streaming, async execution, and observability without bolted-on workarounds. This post maps the old mental model to the new one and shows you what to actually use today.

![LangChain architecture diagram showing chain composition](https://python.langchain.com/img/langchain_stack.png)

## The Old Model: Chains as Class Hierarchies

In the original LangChain design, everything was a chain subclass. `LLMChain`, `RetrievalQA`, `ConversationalRetrievalChain` — each was a class with a `run()` or `__call__()` method. Composing them meant passing chain instances as arguments to other chain constructors.

```python
# Old API (still works but deprecated)
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a one-sentence summary of {topic}."
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("quantum computing")
print(result)
```

This worked, but it had problems:
- No native streaming — you had to implement callbacks manually
- No async support without subclassing
- Composition between chain types was inconsistent
- Observability required injecting callback handlers everywhere

## The New Model: Everything Is a Runnable

LangChain's new foundation is the `Runnable` protocol. Any object that implements `invoke()`, `stream()`, `batch()`, and their async counterparts is a Runnable. Prompts, LLMs, output parsers, retrievers — they all implement this interface.

```python
from langchain_core.runnables import Runnable
```

The three core methods you need to know:

| Method | Synchronous | Asynchronous |
|---|---|---|
| Single call | `invoke(input)` | `ainvoke(input)` |
| Streaming | `stream(input)` | `astream(input)` |
| Batched | `batch(inputs)` | `abatch(inputs)` |

Every LangChain component — `ChatOpenAI`, `ChatPromptTemplate`, `StrOutputParser` — implements all of these. This uniformity is what makes LCEL composition possible.

## LCEL: The Pipe Operator for Chain Composition

LangChain Expression Language uses Python's `|` operator (the bitwise OR, overloaded via `__or__`) to compose Runnables into pipelines. The result is a new Runnable that chains the components left-to-right.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Define components
prompt = ChatPromptTemplate.from_template(
    "Explain {concept} in one paragraph, as if to a software engineer."
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
parser = StrOutputParser()

# Compose with pipe operator
chain = prompt | llm | parser

# Invoke
result = chain.invoke({"concept": "attention mechanisms"})
print(result)
```

What `|` does under the hood: it creates a `RunnableSequence`. Calling `invoke()` on the sequence calls each component's `invoke()` in order, passing the output of one as the input to the next. The type transformations at each step:

- `ChatPromptTemplate.invoke(dict)` → `ChatPromptValue` (list of messages)
- `ChatOpenAI.invoke(list[BaseMessage])` → `AIMessage`
- `StrOutputParser.invoke(AIMessage)` → `str`

This is not magic — it's the same data flowing through sequential `.invoke()` calls, but wrapped in a composable abstraction.

## Streaming with LCEL

Streaming is where LCEL really earns its keep. Because every component implements `stream()`, the pipeline can propagate streaming from the LLM all the way to your output:

```python
# Streaming output token by token
for chunk in chain.stream({"concept": "gradient descent"}):
    print(chunk, end="", flush=True)
print()  # newline at end
```

This works because `RunnableSequence.stream()` calls each component in turn, and `ChatOpenAI.stream()` yields tokens as they arrive from the API. Components that don't produce streamed output (like `ChatPromptTemplate`) pass through transparently.

```python
# Async streaming — critical for FastAPI/async web servers
import asyncio

async def stream_response():
    async for chunk in chain.astream({"concept": "backpropagation"}):
        print(chunk, end="", flush=True)
    print()

asyncio.run(stream_response())
```

## Parallel Execution with RunnableParallel

LCEL also supports parallel execution of multiple branches with `RunnableParallel`:

```python
from langchain_core.runnables import RunnableParallel

# Two chains running in parallel on the same input
joke_chain = (
    ChatPromptTemplate.from_template("Tell a short joke about {topic}.")
    | llm
    | StrOutputParser()
)

fact_chain = (
    ChatPromptTemplate.from_template("Share one surprising fact about {topic}.")
    | llm
    | StrOutputParser()
)

combined = RunnableParallel(
    joke=joke_chain,
    fact=fact_chain
)

result = combined.invoke({"topic": "neural networks"})
print(result["joke"])
print(result["fact"])
```

The dict syntax `{"key": runnable}` is shorthand for `RunnableParallel`. LangChain automatically runs the branches concurrently using threads (sync) or asyncio tasks (async).

## Passing Through Data with RunnablePassthrough

A common pattern: you need to pass the original input through alongside the LLM output (e.g., for RAG pipelines where you need both the retrieved context and the question):

```python
from langchain_core.runnables import RunnablePassthrough

# RAG-style: pass question through while also retrieving documents
retriever = vectorstore.as_retriever()

rag_chain = (
    RunnableParallel(
        context=retriever,
        question=RunnablePassthrough()
    )
    | ChatPromptTemplate.from_template(
        "Answer based on context:\n\nContext: {context}\n\nQuestion: {question}"
    )
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is gradient clipping?")
```

`RunnablePassthrough()` simply returns its input unchanged. Combined with `RunnableParallel`, it lets you build pipelines that carry data alongside transformations.

## Configurable Runnables

One of LCEL's underappreciated features: you can make components configurable at runtime without rebuilding the chain:

```python
from langchain_core.runnables import ConfigurableField

llm = ChatOpenAI(model="gpt-4o-mini").configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="Controls randomness of the output"
    )
)

chain = prompt | llm | parser

# Use default temperature
result_default = chain.invoke({"concept": "entropy"})

# Override temperature at runtime
result_creative = chain.with_config(
    configurable={"llm_temperature": 1.2}
).invoke({"concept": "entropy"})
```

This is useful when you want a single chain definition but need to vary behavior per request — for example, deterministic output for structured extraction and creative output for text generation.

![LangChain LCEL pipeline showing data flow between components](https://python.langchain.com/assets/images/langchain_stack_062024-9b7bbc1e3c51b7ae8a2cd68a6f3bfd49.jpg)

## Inspecting the Chain: The Introspection API

LCEL chains are transparent — you can inspect them programmatically:

```python
# Get the input schema
print(chain.input_schema.schema())
# {'title': 'PromptInput', 'type': 'object', 'properties': {'concept': {'title': 'Concept', 'type': 'string'}}}

# Get the output schema
print(chain.output_schema.schema())
# {'title': 'StrOutputParserOutput', 'type': 'string'}

# Visualize the chain (requires graphviz)
chain.get_graph().print_ascii()
```

This introspection enables LangSmith to automatically trace every component in the chain without you adding any instrumentation code.

## Migration Guide: Old API → LCEL

Here's the mapping from common legacy patterns to their LCEL equivalents:

```python
# OLD: LLMChain
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(topic="AI")

# NEW: LCEL
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "AI"})

# OLD: RetrievalQA
from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
result = qa.run("What is RAG?")

# NEW: LCEL RAG
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
result = chain.invoke("What is RAG?")

# OLD: ConversationChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
conv = ConversationChain(llm=llm, memory=ConversationBufferMemory())
conv.predict(input="Hello")

# NEW: Manual message history management (or use RunnableWithMessageHistory)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history=lambda session_id: ChatMessageHistory(),
    input_messages_key="question",
    history_messages_key="history"
)
```

## Package Structure: What to Import From Where

LangChain split into multiple packages in 2024. Import from the right package or you'll get deprecation warnings:

```python
# Core abstractions (stable, rarely changes)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Provider integrations (install separately)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader

# High-level chains and agents (uses langchain_core internally)
from langchain.chains import create_retrieval_chain
from langchain.agents import create_tool_calling_agent, AgentExecutor
```

The key insight: `langchain_core` is the stable foundation. `langchain_openai`, `langchain_anthropic`, etc. are provider-specific packages you install only when needed. `langchain` itself is now mostly a convenience layer on top of these.

## Conclusion

LangChain's shift to LCEL and the Runnable protocol was the right call. The old class hierarchy made simple things verbose and complex things messy. LCEL's pipe-based composition is learnable in an afternoon, and the uniformity of the Runnable interface means streaming, batching, and async work consistently across every component.

If you're starting a new project, go all-in on LCEL from day one. If you're maintaining legacy code, migrate incrementally — the old chains still work, but you'll want LCEL for anything performance-sensitive or production-facing.
