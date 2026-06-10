"""
backend/groq_client.py
Core backend logic: Groq API client initialisation and streaming chat.
"""

import os
from dotenv import load_dotenv

# Load .env from project root (one level up from backend/)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

try:
    from groq import Groq
except ImportError as exc:
    raise ImportError(
        "The 'groq' package is not installed. "
        "Please run: pip install -r requirements.txt"
    ) from exc

# ── Model configuration ──────────────────────────────────────────────────────
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, direct, and intelligent AI assistant "
    "powered by Groq's ultra-fast LPU inference engine."
)


def get_client() -> Groq:
    """Return an authenticated Groq client."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.startswith("your_"):
        raise EnvironmentError(
            "GROQ_API_KEY is missing or invalid. "
            "Set it in the .env file at the project root."
        )
    return Groq(api_key=api_key)


def stream_chat(
    messages: list[dict], model: str = DEFAULT_MODEL, temperature: float = 0.7
):
    """
    Yield text chunks from the Groq streaming API.

    Parameters
    ----------
    messages    : Conversation history in OpenAI message format.
    model       : Groq model identifier.
    temperature : Sampling temperature (0‒2).

    Yields
    ------
    str — incremental response text chunks.
    """
    client = get_client()
    stream = client.chat.completions.create(
        messages=messages,
        model=model,
        stream=True,
        temperature=temperature,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta is not None:
            yield delta


def build_system_message(content: str = DEFAULT_SYSTEM_PROMPT) -> dict:
    """Return a system-role message dict."""
    return {"role": "system", "content": content}


def build_user_message(content: str) -> dict:
    """Return a user-role message dict."""
    return {"role": "user", "content": content}


def build_assistant_message(content: str) -> dict:
    """Return an assistant-role message dict."""
    return {"role": "assistant", "content": content}
