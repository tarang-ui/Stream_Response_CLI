"""
tests/test_groq_client.py
Basic unit tests for backend/groq_client.py (no live API calls).
"""

import os
import sys
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.groq_client import (
    build_system_message,
    build_user_message,
    build_assistant_message,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
)


def test_build_system_message_role():
    msg = build_system_message("Be helpful.")
    assert msg["role"] == "system"
    assert msg["content"] == "Be helpful."


def test_build_user_message_role():
    msg = build_user_message("Hello!")
    assert msg["role"] == "user"
    assert msg["content"] == "Hello!"


def test_build_assistant_message_role():
    msg = build_assistant_message("Hi there!")
    assert msg["role"] == "assistant"
    assert msg["content"] == "Hi there!"


def test_default_model_is_string():
    assert isinstance(DEFAULT_MODEL, str) and len(DEFAULT_MODEL) > 0


def test_default_system_prompt_is_string():
    assert isinstance(DEFAULT_SYSTEM_PROMPT, str) and len(DEFAULT_SYSTEM_PROMPT) > 0


def test_get_client_raises_without_key(monkeypatch):
    """get_client() should raise EnvironmentError when key is absent."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from backend import groq_client
    with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
        groq_client.get_client()
