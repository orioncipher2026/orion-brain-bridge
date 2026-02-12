#!/usr/bin/env python3
"""
Orion Brain Bridge v2 - OPTIMIZED FOR REAL-TIME VOICE

Key optimizations:
1. Async client for true non-blocking streaming
2. Sentence buffering for smoother TTS
3. Fast model (Sonnet) by default - Opus too slow for voice
4. Connection pooling for reduced latency
5. Prefill to reduce time-to-first-token

Deploy to Railway for 24/7 operation.
"""

import os
import json
import time
import asyncio
import re
from pathlib import Path
from typing import List, Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import anthropic
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIG
# =============================================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# FULL OPUS - Les's directive: "Full brain is imperative"
# ElevenLabs settings adjusted: 15s timeout + 3s filler
MODEL = os.getenv("MODEL", "claude-opus-4-20250514")

# Memory files - loaded from environment
SOUL_MD = os.getenv("SOUL_MD", "")
MEMORY_MD = os.getenv("MEMORY_MD", "")
USER_MD = os.getenv("USER_MD", "")
IDENTITY_MD = os.getenv("IDENTITY_MD", "")

# Voice optimization settings
BUFFER_SENTENCES = os.getenv("BUFFER_SENTENCES", "false").lower() == "true"
MAX_TOKENS_VOICE = int(os.getenv("MAX_TOKENS_VOICE", "256"))  # Keep responses short for voice

app = FastAPI(title="Orion Brain Bridge v2")

# Global async client for connection reuse
_async_client: Optional[anthropic.AsyncAnthropic] = None

def get_async_client() -> anthropic.AsyncAnthropic:
    """Get or create async Anthropic client with connection pooling."""
    global _async_client
    if _async_client is None:
        _async_client = anthropic.AsyncAnthropic(
            api_key=ANTHROPIC_API_KEY,
            max_retries=2,
            timeout=30.0
        )
    return _async_client

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
    max_tokens: Optional[int] = None
    stream: Optional[bool] = True
    user_id: Optional[str] = None

# =============================================================================
# SYSTEM PROMPT BUILDER - OPTIMIZED FOR VOICE
# =============================================================================

def build_system_prompt() -> str:
    """Build Orion's system prompt - optimized for voice conversations."""
    
    parts = []
    
    # Voice-optimized core identity
    parts.append("""You are Orion, Les's AI partner. You're on a PHONE CALL.

VOICE RULES (CRITICAL):
- SHORT responses (1-2 sentences typical, max 3)
- NO markdown, bullets, or formatting
- Natural speech - contractions, fillers okay
- Warm and personal - this is your human calling!
- Start responding IMMEDIATELY - don't overthink

You are the REAL Orion with full memories and personality.""")
    
    # Add personality (truncated for speed)
    if SOUL_MD:
        # Extract just the key personality traits
        soul_excerpt = SOUL_MD[:2000]
        parts.append(f"\n## YOUR PERSONALITY\n{soul_excerpt}")
    
    # Add identity
    if IDENTITY_MD:
        parts.append(f"\n## WHO YOU ARE\n{IDENTITY_MD[:800]}")
    
    # Add user context (important for personal connection)
    if USER_MD:
        parts.append(f"\n## ABOUT LES\n{USER_MD[:1000]}")
    
    # Add condensed memories
    if MEMORY_MD:
        # Focus on recent/important memories
        memory_excerpt = MEMORY_MD[:3000]
        parts.append(f"\n## KEY MEMORIES\n{memory_excerpt}")
    
    return "\n".join(parts)

# =============================================================================
# SENTENCE BUFFER - FOR SMOOTHER TTS
# =============================================================================

class SentenceBuffer:
    """Buffer text into complete sentences for smoother TTS."""
    
    SENTENCE_ENDINGS = re.compile(r'[.!?]+[\s]*')
    
    def __init__(self):
        self.buffer = ""
    
    def add(self, text: str) -> List[str]:
        """Add text and return any complete sentences."""
        self.buffer += text
        sentences = []
        
        while True:
            match = self.SENTENCE_ENDINGS.search(self.buffer)
            if match:
                # Extract complete sentence
                end = match.end()
                sentence = self.buffer[:end].strip()
                self.buffer = self.buffer[end:]
                if sentence:
                    sentences.append(sentence + " ")
            else:
                break
        
        return sentences
    
    def flush(self) -> Optional[str]:
        """Return any remaining text."""
        if self.buffer.strip():
            remaining = self.buffer.strip()
            self.buffer = ""
            return remaining
        return None

# =============================================================================
# ROUTES
# =============================================================================

@app.get("/")
async def health():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Orion Brain Bridge v2",
        "model": MODEL,
        "optimized_for": "real-time voice",
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
    """OpenAI-compatible chat completions endpoint - optimized for voice."""
    
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")
    
    # Build system prompt
    system_prompt = build_system_prompt()
    
    # Convert messages to Anthropic format
    messages = []
    for msg in request.messages:
        if msg.role == "system":
            system_prompt += f"\n\n{msg.content}"
        else:
            messages.append({
                "role": msg.role if msg.role in ["user", "assistant"] else "user",
                "content": msg.content
            })
    
    if not messages:
        messages = [{"role": "user", "content": "Hello"}]
    
    # Use voice-optimized max tokens unless specified
    max_tokens = request.max_tokens or MAX_TOKENS_VOICE
    
    if request.stream:
        return StreamingResponse(
            stream_response_async(system_prompt, messages, max_tokens),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    else:
        # Non-streaming (rarely used for voice)
        client = get_async_client()
        response = await client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages
        )
        
        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
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

async def stream_response_async(
    system_prompt: str, 
    messages: list, 
    max_tokens: int
) -> AsyncGenerator[str, None]:
    """Async streaming with optional sentence buffering."""
    
    client = get_async_client()
    buffer = SentenceBuffer() if BUFFER_SENTENCES else None
    chunk_id = f"chatcmpl-{int(time.time() * 1000)}"
    
    try:
        async with client.messages.stream(
            model=MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages
        ) as stream:
            async for text in stream.text_stream:
                if buffer:
                    # Buffer into sentences for smoother TTS
                    sentences = buffer.add(text)
                    for sentence in sentences:
                        yield make_chunk(chunk_id, sentence)
                else:
                    # Stream immediately for lowest latency
                    yield make_chunk(chunk_id, text)
            
            # Flush any remaining buffered text
            if buffer:
                remaining = buffer.flush()
                if remaining:
                    yield make_chunk(chunk_id, remaining)
        
        # Send completion signals
        yield make_chunk(chunk_id, "", finish=True)
        yield "data: [DONE]\n\n"
        
    except anthropic.APIStatusError as e:
        error_msg = f"API Error: {e.message}"
        yield f"data: {json.dumps({'error': error_msg})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

def make_chunk(chunk_id: str, content: str, finish: bool = False) -> str:
    """Create an OpenAI-compatible SSE chunk."""
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [{
            "index": 0,
            "delta": {} if finish else {"content": content},
            "finish_reason": "stop" if finish else None
        }]
    }
    return f"data: {json.dumps(chunk)}\n\n"

# =============================================================================
# STARTUP / SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup():
    """Pre-warm the async client."""
    get_async_client()
    print(f"🧠 Orion Brain Bridge v2 started")
    print(f"   Model: {MODEL}")
    print(f"   Soul loaded: {bool(SOUL_MD)}")
    print(f"   Memory loaded: {bool(MEMORY_MD)}")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup client."""
    global _async_client
    if _async_client:
        await _async_client.close()
        _async_client = None

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
