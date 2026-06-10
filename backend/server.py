"""
backend/server.py
FastAPI backend server for the Real-Time Streaming Chatbot.

Endpoints:
  GET  /health          — health + API key status check
  GET  /models          — list available Groq models
  POST /chat            — non-streaming single response
  POST /chat/stream     — Server-Sent Events (SSE) streaming response

Run with:
  uvicorn backend.server:app --reload --port 8000
  (from the project root)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import AsyncGenerator

# ── Ensure project root is importable ────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from backend.groq_client import (  # noqa: E402
    get_client,
    stream_chat,
    build_system_message,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Groq Streaming Chatbot API",
    description="Backend API powering the real-time streaming chatbot via Groq LPU inference.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow Streamlit (localhost:8501) and any other local origin during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Available models ──────────────────────────────────────────────────────────
AVAILABLE_MODELS: list[str] = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# ── Request / Response schemas ────────────────────────────────────────────────


class Message(BaseModel):
    role: str = Field(..., examples=["user"])
    content: str = Field(..., examples=["Hello!"])


class ChatRequest(BaseModel):
    messages: list[Message] = Field(
        default_factory=list,
        description="Conversation history (excluding system message).",
    )
    model: str = Field(default=DEFAULT_MODEL, examples=[DEFAULT_MODEL])
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT)


class ChatResponse(BaseModel):
    reply: str
    model: str
    turns: int


# ── Helper ───────────────────────────────────────────────────────────────────


def _build_messages(req: ChatRequest) -> list[dict]:
    """Prepend system message to conversation history."""
    history = [build_system_message(req.system_prompt)]
    history += [{"role": m.role, "content": m.content} for m in req.messages]
    return history


async def _sse_generator(req: ChatRequest) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted chunks from the Groq stream."""
    loop = asyncio.get_event_loop()
    full_response = ""

    # Run the synchronous generator in a thread pool to avoid blocking
    def _collect():
        return list(
            stream_chat(
                messages=_build_messages(req),
                model=req.model,
                temperature=req.temperature,
            )
        )

    try:
        chunks = await loop.run_in_executor(None, _collect)
        for chunk in chunks:
            full_response += chunk
            data = json.dumps({"chunk": chunk, "done": False})
            yield f"data: {data}\n\n"

        # Final done event
        done_data = json.dumps(
            {"chunk": "", "done": True, "full_response": full_response}
        )
        yield f"data: {done_data}\n\n"

    except EnvironmentError as exc:
        error_data = json.dumps({"error": str(exc), "done": True})
        yield f"data: {error_data}\n\n"
    except Exception as exc:
        error_data = json.dumps({"error": f"API error: {str(exc)}", "done": True})
        yield f"data: {error_data}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["System"])
async def health_check():
    """Check server health and Groq API key validity."""
    try:
        get_client()
        api_status = "connected"
    except EnvironmentError as e:
        api_status = f"error: {e}"

    return {
        "status": "ok",
        "api_key_status": api_status,
        "default_model": DEFAULT_MODEL,
    }


@app.get("/models", tags=["System"])
async def list_models():
    """Return the list of supported Groq models."""
    return {"models": AVAILABLE_MODELS, "default": DEFAULT_MODEL}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """
    Non-streaming chat endpoint.
    Collects the full Groq response before returning.
    """
    try:
        get_client()  # validates key early
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    loop = asyncio.get_event_loop()

    def _run():
        return "".join(
            stream_chat(
                messages=_build_messages(req),
                model=req.model,
                temperature=req.temperature,
            )
        )

    try:
        reply = await loop.run_in_executor(None, _run)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Groq API error: {exc}")

    return ChatResponse(
        reply=reply,
        model=req.model,
        turns=len(req.messages) + 1,
    )


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(req: ChatRequest):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Each event: data: {"chunk": "...", "done": false}
    Final event: data: {"chunk": "", "done": true, "full_response": "..."}
    """
    try:
        get_client()
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return StreamingResponse(
        _sse_generator(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Dev entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["."],
    )
