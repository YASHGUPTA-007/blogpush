---
title: >-
  LangChain Prompt Templates and Output Parsers: PromptTemplate,
  ChatPromptTemplate, and Pydantic Parsers
excerpt: >-
  Prompt templates and output parsers are the input and output contracts of your
  LLM pipeline. Build them wrong and your chain breaks on every edge case.
author: Soham Sharma
authorName: Soham Sharma
category: AI
tags:
  - LangChain
  - Prompt Engineering
  - Pydantic
  - Output Parsing
  - LLM
status: published
featuredImage: >-
  https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=1200&auto=format&fit=crop&q=80
colab_notebook: >-
  https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/langchain/langchain_2.ipynb
series_id: langchain-production
series_slug: langchain-production
series_title: LangChain / LangSmith / LangGraph — In Production
difficulty: beginner
week: null
day: 7
tools:
  - LangChain
  - Pydantic
  - OpenAI
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YASHGUPTA-007/blogpush/blob/main/notebooks/langchain/langchain_2.ipynb)


An LLM call without a properly structured prompt is a function with no contract. You get output, but you can't guarantee its format, and your parser will fail on inputs you didn't test. LangChain's prompt templates and output parsers are the tooling that turns "call an LLM" into "call an LLM and get back a typed Python object." This post covers both ends of the pipe.

## PromptTemplate: Single-String Inputs

`PromptTemplate` is for non-chat models — older completion APIs that take a single string input. It uses Python `str.format`-style variable substitution.

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["topic", "audience"],
    template="Write a one-sentence explanation of {topic} for {audience}."
)

# Format the prompt
formatted = template.format(topic="gradient descent", audience="high school students")
print(formatted)
print(type(formatted))
```

**Output:**
```text
Write a one-sentence explanation of gradient descent for high school students.
<class 'str'>
```

`PromptTemplate.format()` returns a plain string. For chat models, use `ChatPromptTemplate` instead — it returns a list of typed messages.

### from_template: the shorthand constructor

```python
from langchain_core.prompts import PromptTemplate

# Variables auto-extracted from the template string
template = PromptTemplate.from_template(
    "Summarize this text in {num_sentences} sentences:\n\n{text}"
)

print(template.input_variables)
formatted = template.format(num_sentences=2, text="LangChain is a framework for building LLM-powered applications. It provides abstractions for prompts, chains, agents, and memory.")
print(formatted)
```

**Output:**
```text
['num_sentences', 'text']
Summarize this text in 2 sentences:

LangChain is a framework for building LLM-powered applications. It provides abstractions for prompts, chains, agents, and memory.
```

`from_template` parses the `{variable}` placeholders automatically — no need to list `input_variables` manually.

## ChatPromptTemplate: Structured Multi-Turn Prompts

Chat models (GPT-4, Claude, Gemini) expect a list of messages with roles. `ChatPromptTemplate` is the correct abstraction for these models.

```python
from langchain_core.prompts import ChatPromptTemplate

# Shorthand tuple syntax: (role, template_string)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Always respond in {language}."),
    ("human", "{question}"),
])

messages = prompt.format_messages(
    role="senior Python engineer",
    language="English",
    question="What is the GIL?"
)

for msg in messages:
    print(f"[{msg.type}] {msg.content}")
```

**Output:**
```text
[system] You are a senior Python engineer. Always respond in English.
[human] What is the GIL?
```

The `format_messages()` method returns a list of `BaseMessage` objects. These are what chat models receive.

### Including conversation history with MessagesPlaceholder

For multi-turn conversations, you need a placeholder to inject previous messages dynamically:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Simulate previous conversation
history = [
    HumanMessage(content="What is PyTorch?"),
    AIMessage(content="PyTorch is an open-source deep learning framework developed by Meta."),
]

messages = prompt.format_messages(
    chat_history=history,
    input="How does it compare to TensorFlow?"
)

for msg in messages:
    print(f"[{msg.type}]: {msg.content[:60]}...")
```

**Output:**
```text
[system]: You are a helpful assistant....
[human]: What is PyTorch?...
[ai]: PyTorch is an open-source deep learning framework develo...
[human]: How does it compare to TensorFlow?...
```

`MessagesPlaceholder` is the correct way to inject dynamic message history — don't concatenate message lists manually.

![LLM prompt template structure diagram](https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=1200&auto=format&fit=crop&q=80)

## Partial Templates: Pre-filling Variables

`partial()` lets you fix some variables upfront and leave others for later. This is useful for injecting system-level context (API version, date, user role) into a reusable prompt.

```python
from langchain_core.prompts import ChatPromptTemplate
from datetime import date

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for {company}. Today's date is {today}. Help with: {task_type}"),
    ("human", "{user_input}"),
])

# Partially fill system-level variables
company_prompt = prompt.partial(
    company="BotMartz",
    today=str(date.today()),
    task_type="technical questions",
)

# Later, fill user-specific variables
messages = company_prompt.format_messages(user_input="What is LCEL?")
print(messages[0].content)
print(messages[1].content)
```

**Output:**
```text
You are an assistant for BotMartz. Today's date is 2025-04-17. Help with: technical questions
What is LCEL?
```

`partial()` returns a new prompt template with the specified variables pre-filled. This lets you define a base prompt once and create specialized versions without duplicating the template string.

## Output Parsers

Output parsers take the raw `AIMessage` response and transform it into a typed Python object. Without parsers, you're string-splitting LLM output manually — which breaks as soon as the model slightly changes its formatting.

### StrOutputParser: The Simplest Parser

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("Name one use case for {technology}.")
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)

result = chain.invoke({"technology": "vector databases"})
print(type(result))
print(result)
```

**Output:**
```text
<class 'str'>
One use case for vector databases is semantic search, where user queries are converted to embeddings and matched against document embeddings to find contextually relevant results rather than keyword matches.
```

`StrOutputParser` simply extracts `.content` from the `AIMessage`. It's the right choice when you need the raw text and will handle structure yourself.

### JsonOutputParser: Structured Dict Output

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Always respond with valid JSON only. No prose."),
        ("human", "Return a JSON object with keys 'name', 'founded', and 'known_for' for: {company}"),
    ])
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | parser
)

result = chain.invoke({"company": "OpenAI"})
print(type(result))
print(result)
```

**Output:**
```text
<class 'dict'>
{'name': 'OpenAI', 'founded': 2015, 'known_for': 'Developing advanced AI systems including GPT-4 and ChatGPT'}
```

> Note: LLM output content may vary slightly by run.

`JsonOutputParser` handles markdown-fenced JSON (` ```json ... ``` `) automatically — if the model wraps its response in code fences, the parser strips them.

### PydanticOutputParser: Typed, Validated Output

This is the production-grade parser. You define a Pydantic model, and LangChain generates format instructions, parses the JSON, validates it against the schema, and returns a typed Python object.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class ModelCard(BaseModel):
    name: str = Field(description="Full model name")
    developer: str = Field(description="Organization that created the model")
    release_year: int = Field(description="Year of public release")
    capabilities: List[str] = Field(description="List of 3 key capabilities")
    context_window: int = Field(description="Context window size in tokens")

parser = PydanticOutputParser(pydantic_object=ModelCard)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical AI researcher. Return information as JSON.\n{format_instructions}"),
    ("human", "Create a model card for: {model_name}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | parser

result = chain.invoke({"model_name": "GPT-4"})
print(type(result))
print(f"Name: {result.name}")
print(f"Developer: {result.developer}")
print(f"Release year: {result.release_year}")
print(f"Context window: {result.context_window:,} tokens")
print(f"Capabilities: {result.capabilities}")
```

**Output:**
```text
<class '__main__.ModelCard'>
Name: GPT-4
Developer: OpenAI
Release year: 2023
Context window: 128,000 tokens
Capabilities: ['Advanced reasoning', 'Multimodal understanding', 'Code generation and debugging']
```

> Note: LLM outputs may vary.

The returned object is a proper `ModelCard` instance — you get type hints, `.model_dump()`, IDE autocomplete, and Pydantic validation all for free.

### What format_instructions generates

It's worth seeing what `parser.get_format_instructions()` actually produces:

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Simple(BaseModel):
    score: int = Field(description="Score from 1-10")
    reason: str = Field(description="One-sentence explanation")

parser = PydanticOutputParser(pydantic_object=Simple)
print(parser.get_format_instructions())
```

**Output:**
```text
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"properties": {"score": {"description": "Score from 1-10", "title": "Score", "type": "integer"}, "reason": {"description": "One-sentence explanation", "title": "Reason", "type": "string"}}, "required": ["score", "reason"]}
```
```

LangChain generates a schema description and injects it into your system prompt automatically. The model uses this to produce compliant JSON.

## Handling Parser Failures: OutputFixingParser

LLMs occasionally return malformed JSON — extra prose, trailing commas, wrong types. `OutputFixingParser` wraps any parser and automatically calls the LLM a second time to fix parse failures.

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from pydantic import BaseModel

class Rating(BaseModel):
    score: int
    comment: str

base_parser = PydanticOutputParser(pydantic_object=Rating)

# Wrap with the fixing parser
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
)

# Simulate malformed output from an LLM
malformed_output = '{"score": "eight", "comment": "Great product"}'

try:
    base_parser.parse(malformed_output)
except Exception as e:
    print(f"Base parser failed: {type(e).__name__}")

# OutputFixingParser calls LLM to repair
fixed = fixing_parser.parse(malformed_output)
print(f"Fixed result: score={fixed.score}, comment={fixed.comment}")
print(f"Type: {type(fixed.score)}")
```

**Output:**
```text
Base parser failed: ValidationError
Fixed result: score=8, comment=Great product
Type: <class 'int'>
```

`OutputFixingParser` sends the malformed output back to the LLM with a correction prompt. `"eight"` → `8` as an int. Use this when you can't control the model producing the output (e.g., you're parsing outputs from a less capable model).

### Gotcha: partial variable injection order matters

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "{question}"),
])

# Wrong: partial a variable that doesn't exist
try:
    bad = prompt.partial(nonexistent_var="value")
    bad.format_messages(role="assistant", question="hi")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Correct
good = prompt.partial(role="Python expert")
result = good.format_messages(question="What is a decorator?")
print(result[0].content)
```

**Output (raises):**
```text
Error: KeyError: 'nonexistent_var'
```

**Output:**
```text
You are a Python expert.
```

`partial()` does not validate variable names at call time — the `KeyError` surfaces when you call `format_messages()`. Always check `prompt.input_variables` before calling `partial()` with dynamic variable names.

![Output parsing pipeline from raw LLM text to structured Python objects](https://images.unsplash.com/photo-1639762681485-074b7f938ba0?w=1200&auto=format&fit=crop&q=80)

## CommaSeparatedListOutputParser: Quick Lists

For simple list outputs, `CommaSeparatedListOutputParser` is faster to set up than Pydantic:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

chain = (
    ChatPromptTemplate.from_messages([
        ("system", "{format_instructions}"),
        ("human", "List 5 Python web frameworks."),
    ]).partial(format_instructions=parser.get_format_instructions())
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | parser
)

result = chain.invoke({})
print(type(result))
print(result)
```

**Output:**
```text
<class 'list'>
['Django', 'Flask', 'FastAPI', 'Tornado', 'Starlette']
```

Useful for quick extraction, but Pydantic is better for anything that needs validation or nested structure.

## Conclusion

Prompt templates and output parsers are the contract layer of your LLM pipeline. `ChatPromptTemplate` with `MessagesPlaceholder` handles multi-turn conversations cleanly. `partial()` pre-fills shared context so you don't repeat yourself. `PydanticOutputParser` turns unstructured LLM text into validated, typed Python objects — and `OutputFixingParser` handles the cases where the model doesn't cooperate. Build these correctly upfront and you won't be debugging malformed JSON at 2am.

The next post covers working with LLMs and chat models — OpenAI, Anthropic, and local models via Ollama — and how to swap models without changing your chain.
