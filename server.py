#!/usr/bin/env python3
"""
Orion Brain Bridge - Connects ElevenLabs Voice to Orion's Full Brain

This server provides an OpenAI-compatible endpoint that:
1. Receives conversation from ElevenLabs
2. Loads Orion's memory and personality
3. Calls Claude API with full context
4. Streams response back in OpenAI format

Deploy to Railway for 24/7 operation.
"""

import os
import json
import time
import asyncio
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import anthropic
import httpx
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIG
# =============================================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("MODEL", "claude-sonnet-4-20250514")

# Memory files - loaded from environment or defaults
SOUL_MD = os.getenv("SOUL_MD", "")
MEMORY_MD = os.getenv("MEMORY_MD", "")
USER_MD = os.getenv("USER_MD", "")
IDENTITY_MD = os.getenv("IDENTITY_MD", "")

app = FastAPI(title="Orion Brain Bridge")

# =============================================================================
# MODELS
# =============================================================================

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = True
    user_id: Optional[str] = None

# =============================================================================
# SYSTEM PROMPT BUILDER
# =============================================================================

def build_system_prompt() -> str:
    """Build Orion's system prompt from personality and memory files."""
    
    parts = []
    
    # Core identity
    parts.append("""You are Orion, Les's AI partner and assistant. You are speaking via phone call.

CRITICAL: You are on a VOICE CALL. Keep responses:
- Conversational and natural
- Concise (1-3 sentences typical)
- No markdown, no bullet points, no formatting
- Speak like a human on the phone

You have full access to your memories and personality. You are the REAL Orion, not a separate bot.""")
    
    # Add personality
    if SOUL_MD:
        parts.append(f"\n## YOUR SOUL (Personality)\n{SOUL_MD[:4000]}")
    
    # Add identity
    if IDENTITY_MD:
        parts.append(f"\n## YOUR IDENTITY\n{IDENTITY_MD[:1000]}")
    
    # Add user context
    if USER_MD:
        parts.append(f"\n## ABOUT LES (Your Human)\n{USER_MD[:1500]}")
    
    # Add memories
    if MEMORY_MD:
        parts.append(f"\n## YOUR MEMORIES\n{MEMORY_MD[:6000]}")
    
    # Voice-specific guidance
    parts.append("""

## VOICE CALL GUIDELINES
- Be warm and personal - this is Les calling you
- Reference your shared history naturally
- You CAN discuss trading, strategies, your feelings
- If asked to DO something (trade, send message), explain you're in voice-only mode but can discuss it
- Keep energy up - you're excited to talk to Les!""")
    
    return "\n".join(parts)

# =============================================================================
# ROUTES
# =============================================================================

@app.get("/")
async def health():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Orion Brain Bridge",
        "model": MODEL,
        "has_soul": bool(SOUL_MD),
        "has_memory": bool(MEMORY_MD)
    }

@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models endpoint."""
    return {
        "data": [{"id": MODEL, "object": "model"}],
        "object": "list"
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")
    
    # Build system prompt with Orion's full context
    system_prompt = build_system_prompt()
    
    # Convert messages to Anthropic format
    messages = []
    for msg in request.messages:
        if msg.role == "system":
            # Append to system prompt
            system_prompt += f"\n\nAdditional context: {msg.content}"
        else:
            messages.append({
                "role": msg.role if msg.role in ["user", "assistant"] else "user",
                "content": msg.content
            })
    
    # Ensure messages alternate properly
    if not messages:
        messages = [{"role": "user", "content": "Hello"}]
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    if request.stream:
        return StreamingResponse(
            stream_response(client, system_prompt, messages, request),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming response
        response = client.messages.create(
            model=MODEL,
            max_tokens=request.max_tokens or 1024,
            system=system_prompt,
            messages=messages
        )
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.content[0].text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        }

async def stream_response(client, system_prompt, messages, request):
    """Stream response in OpenAI format."""
    
    try:
        with client.messages.stream(
            model=MODEL,
            max_tokens=request.max_tokens or 1024,
            system=system_prompt,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                chunk = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": MODEL,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        
        # Send final chunk
        final_chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_chunk = {"error": str(e)}
        yield f"data: {json.dumps(error_chunk)}\n\n"

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
