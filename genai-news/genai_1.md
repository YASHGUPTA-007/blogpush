---
title: "GPT-4o Multimodal Capabilities: What Developers Can Actually Build Today"
excerpt: "GPT-4o's multimodal API is available and production-ready. Here's a clear-eyed look at what works, what breaks, and what you can ship right now."
author: "Soham Sharma"
category: "AI"
tags: ["GPT-4o", "Multimodal", "OpenAI", "LLM", "GenAI"]
status: "published"
featuredImage: "https://images.unsplash.com/photo-1686191128892-3b37add4c844?w=1200&auto=format&fit=crop&q=80"
---

GPT-4o ("omni") unified text, image, and audio in a single model endpoint. The marketing narrative was about natural conversation with emotional voice responses. The engineering reality is more interesting: a production API that handles vision, structured JSON extraction from images, and document understanding at latencies and costs that make real applications viable. This post is the pragmatic developer's guide — what you can build today, where the model actually struggles, and what the benchmarks don't tell you.

![Futuristic AI interface showing multimodal input streams on a holographic display](https://images.unsplash.com/photo-1686191128892-3b37add4c844?w=1200&auto=format&fit=crop&q=80)

## What GPT-4o Actually Exposes via API

As of 2025, the GPT-4o API supports three input modalities:

| Modality | API Status | Notes |
|---|---|---|
| Text | GA | Standard chat completions |
| Image | GA | Base64 or URL, up to 20MB |
| Audio input | GA | Whisper-quality STT in the model |
| Audio output | GA | Real-time audio stream via Realtime API |
| Video | Not supported | Must extract frames manually |

The image API works through the standard chat completions endpoint — images are passed as content parts alongside text:

```python
from openai import OpenAI
import base64
from pathlib import Path

client = OpenAI()  # uses OPENAI_API_KEY env var

def encode_image(image_path: str) -> str:
    """Encode local image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_image(image_path: str, question: str) -> str:
    """Send image + question to GPT-4o and return response."""
    base64_image = encode_image(image_path)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"  # or "low" for faster/cheaper, "auto" for model choice
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ],
        max_tokens=1024
    )
    
    return response.choices[0].message.content

# Example usage
result = analyze_image("architecture_diagram.png", 
    "List every component in this system diagram and describe the data flow between them.")
print(result)
```

## Real Use Case 1: Structured Data Extraction from Documents

This is the highest-value use case for most teams today. GPT-4o can extract structured information from PDFs, screenshots, invoices, and forms with accuracy that makes manual processing economically unviable.

```python
from pydantic import BaseModel
from typing import Optional
import json

class InvoiceData(BaseModel):
    vendor_name: str
    invoice_number: str
    invoice_date: str
    due_date: Optional[str]
    subtotal: float
    tax: Optional[float]
    total: float
    line_items: list[dict]

def extract_invoice_data(image_path: str) -> InvoiceData:
    """Extract structured invoice data from an image."""
    base64_image = encode_image(image_path)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Extract invoice data as JSON. Return only valid JSON, no markdown."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                    {
                        "type": "text",
                        "text": f"Extract all invoice fields into this JSON structure: {InvoiceData.schema_json()}"
                    }
                ]
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=2048,
        temperature=0  # Deterministic for extraction tasks
    )
    
    raw_json = response.choices[0].message.content
    return InvoiceData(**json.loads(raw_json))

# The model handles varying formats — handwritten amounts, different date formats, 
# multi-page invoices (if you pass multiple images), etc.
```

**Real-world accuracy**: On standard printed invoices, GPT-4o achieves ~97% field extraction accuracy. On handwritten or low-quality scans, expect 85-92%. The main failure modes: ambiguous numbers (1 vs I, 0 vs O), multi-column tables where column boundaries aren't clear, and amounts without explicit currency symbols.

## Real Use Case 2: Screenshot-to-Code

GPT-4o understands UI screenshots well enough to generate functional HTML/CSS or React components from a mockup. This is genuinely useful for rapid prototyping:

```python
def screenshot_to_code(screenshot_path: str, framework: str = "React") -> str:
    """
    Convert a UI screenshot to code.
    
    Args:
        screenshot_path: Path to UI screenshot
        framework: "React", "HTML", "Tailwind", etc.
    """
    base64_image = encode_image(screenshot_path)
    
    system_prompt = f"""You are an expert {framework} developer.
Convert the UI screenshot into clean, production-quality {framework} code.
Requirements:
- Use semantic HTML
- Include responsive design considerations  
- Use placeholder images and text where appropriate
- Add accessibility attributes (aria-labels, etc.)
Output only the code, no explanation."""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}
                    },
                    {"type": "text", "text": f"Convert this screenshot to {framework} code."}
                ]
            }
        ],
        max_tokens=4096,
        temperature=0.1
    )
    
    return response.choices[0].message.content
```

**What actually works**: Landing pages, dashboards with standard components, data tables, forms. The model produces working code about 70-80% of the time without manual fixes for simple layouts. Complex custom interactions, animations, and precise pixel-level layouts need iteration.

## Real Use Case 3: Multi-Image Reasoning

GPT-4o can reason across multiple images in a single prompt — useful for before/after comparisons, document sequences, or product inspection:

```python
def compare_images(image_paths: list[str], comparison_prompt: str) -> str:
    """
    Reason across multiple images.
    
    Use cases:
    - Compare product photos for QA
    - Analyze before/after medical images
    - Review sequential steps in a process
    """
    content = []
    
    for i, path in enumerate(image_paths):
        content.append({
            "type": "text", 
            "text": f"Image {i+1}:"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(path)}",
                "detail": "high"
            }
        })
    
    content.append({"type": "text", "text": comparison_prompt})
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=2048
    )
    
    return response.choices[0].message.content

# Example: product defect detection
result = compare_images(
    ["product_reference.jpg", "product_sample_1.jpg", "product_sample_2.jpg"],
    "Identify any defects or differences between the reference image and the two samples. List each defect with its location."
)
```

## What Breaks at Scale

Honest assessment of where GPT-4o falls short:

| Limitation | Detail |
|---|---|
| **Fine-grained text in images** | Dense text (contracts, legal docs) at <12pt often misread |
| **Complex tables** | Multi-level headers, merged cells, rotated text fails ~20% |
| **Spatial reasoning** | "How many pixels from the left edge" type questions are unreliable |
| **Consistency** | Same image + same prompt can produce different field values across calls |
| **Cost at scale** | High-detail images: ~$0.003-0.015 per image, adds up fast at 100K+ images/day |
| **Context window** | ~50 images max before quality degrades due to context length |

The consistency issue is the most production-critical. For extraction tasks, always use `temperature=0` and consider running 2-3 extractions and taking the majority vote for high-stakes fields.

## Cost Breakdown

GPT-4o pricing for vision (as of 2025):

```
Low detail mode:  ~85 tokens per image (~$0.000255 at $3/1M input tokens)
High detail mode: Depends on image resolution
  - 512×512:   ~170 tokens
  - 1024×1024: ~765 tokens
  - 2048×2048: ~1105 tokens

A typical invoice (A4, 1200×1700px) ≈ 1200 tokens = ~$0.0036 input
With a 500-token response: ~$0.0041 total per document
```

At 10,000 invoices/day: ~$41/day, ~$1,250/month. Compare that to human data entry at $0.50-2.00 per invoice — the economics are clear for most use cases.

## The Realtime API: Audio in/out

The Realtime API enables genuine low-latency voice interaction (200-300ms) through a WebSocket connection:

```python
import asyncio
import websockets
import json
import base64

async def realtime_voice_session():
    """Simplified Realtime API session setup."""
    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }
    
    async with websockets.connect(url, extra_headers=headers) as ws:
        # Configure session
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "voice": "alloy",              # Voice selection
                "instructions": "You are a helpful assistant.",
                "turn_detection": {
                    "type": "server_vad",      # Server-side VAD
                    "threshold": 0.5
                },
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16"
            }
        }))
        
        # Stream audio chunks as they arrive from microphone
        # (audio_chunk is raw PCM16 bytes from your audio capture)
        audio_chunk = b'\x00' * 3200  # Placeholder — replace with real audio
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio_chunk).decode()
        }))
        
        # Receive responses
        while True:
            message = json.loads(await ws.recv())
            if message["type"] == "response.audio.delta":
                audio_data = base64.b64decode(message["audio"])
                # Send audio_data to your speaker output
            elif message["type"] == "response.done":
                break

# For production, use the official openai Python SDK's Realtime client
# (available in openai >= 1.40.0)
```

## What This Means for You

GPT-4o's multimodal API is mature enough for production use in 2025. The clearest ROI opportunities:

1. **Document processing pipelines** — invoices, receipts, forms, medical records. Replace or augment OCR + parsing with direct extraction.
2. **Visual QA tools** — any workflow where humans currently look at images and make judgments.
3. **Accessibility tooling** — automatic alt-text generation, screen reader content, image description pipelines.
4. **Rapid prototyping** — screenshot-to-code shortens design→implementation cycles.

The areas not ready for unsupervised production use: safety-critical inspection (medical imaging diagnosis, autonomous driving), fine-grained text reading from scanned documents, and anything requiring exact spatial measurements.

The voice API is excellent for demos and consumer apps but introduces platform complexity (WebSocket management, audio streaming, VAD tuning) that most enterprise teams aren't set up for yet.

Start with document extraction if you have unstructured document workflows — it's the most reliable, best-understood use case, and the economics are favorable.

![Person interacting with an AI assistant on a laptop with glowing UI elements](https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=1200&auto=format&fit=crop&q=80)
