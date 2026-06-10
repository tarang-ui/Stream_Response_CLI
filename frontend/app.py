"""
frontend/app.py
Streamlit UI for the Real-Time Streaming Chatbot powered by Groq.
Run with:  streamlit run frontend/app.py
"""

import sys
import os

# Allow imports from project root so backend/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from backend.groq_client import (
    stream_chat,
    build_system_message,
    build_user_message,
    build_assistant_message,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    get_client,
)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Groq Streaming Chatbot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Global dark background ── */
    .stApp {
        background: linear-gradient(135deg, #0d0d1a 0%, #111827 50%, #0a0a16 100%);
        color: #e2e8f0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1a1f2e 100%);
        border-right: 1px solid rgba(99,102,241,0.25);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #a5b4fc;
    }

    /* ── Header banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(99,102,241,0.35);
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .hero-banner p {
        margin: 6px 0 0;
        color: rgba(255,255,255,0.85);
        font-size: 0.95rem;
    }

    /* ── Chat messages ── */
    .chat-bubble-user {
        background: linear-gradient(135deg, #4f46e5, #6366f1);
        border-radius: 18px 18px 4px 18px;
        padding: 14px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        color: #ffffff;
        font-size: 0.95rem;
        box-shadow: 0 4px 16px rgba(99,102,241,0.3);
        line-height: 1.6;
    }
    .chat-bubble-assistant {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 18px 18px 18px 4px;
        backdrop-filter: blur(8px);
        padding: 14px 18px;
        margin: 8px 0;
        max-width: 80%;
        color: #e2e8f0;
        font-size: 0.95rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        line-height: 1.6;
    }
    .role-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
        opacity: 0.7;
    }

    /* ── Input area ── */
    .stTextInput > div > div > input,
    .stTextArea textarea {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(99,102,241,0.4) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
        transition: border-color 0.2s;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
        padding: 10px 24px !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 14px rgba(99,102,241,0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(99,102,241,0.55) !important;
    }

    /* ── Metrics / stat pills ── */
    .stat-pill {
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 50px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 500;
        color: #a5b4fc;
        display: inline-block;
        margin: 2px 4px;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(99,102,241,0.2) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }

    /* ── Selectbox / Slider labels ── */
    label { color: #a5b4fc !important; font-weight: 500 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialisation ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [build_system_message(DEFAULT_SYSTEM_PROMPT)]
if "token_count" not in st.session_state:
    st.session_state.token_count = 0
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Groq Chatbot")
    st.markdown("---")

    # Model selection
    model_choice = st.selectbox(
        "🤖 Model",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        index=0,
    )

    # Temperature slider
    temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.05,
        help="Higher = more creative, Lower = more focused",
    )

    # System prompt
    system_prompt = st.text_area(
        "🧠 System Prompt",
        value=DEFAULT_SYSTEM_PROMPT,
        height=120,
        help="Instructions that define the assistant's persona",
    )

    st.markdown("---")

    # Stats
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    col1.metric("Turns", st.session_state.turn_count)
    col2.metric("Est. Tokens", st.session_state.token_count)

    st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [build_system_message(system_prompt)]
        st.session_state.token_count = 0
        st.session_state.turn_count = 0
        st.rerun()

    # Connection status
    st.markdown("---")
    try:
        get_client()
        st.success("🟢 Groq API Connected", icon="✅")
    except EnvironmentError as e:
        st.error(f"🔴 {e}", icon="❌")

    st.markdown(
        "<small style='color:#64748b;'>Powered by Groq LPU™ Inference</small>",
        unsafe_allow_html=True,
    )


# ── Main chat area ────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <h1>⚡ Real-Time Streaming Chatbot</h1>
        <p>Ultra-fast AI responses powered by Groq's LPU™ inference engine</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render conversation history (skip system message at index 0)
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages[1:]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-bubble-user">'
                f'<div class="role-label">You</div>{msg["content"]}'
                f"</div>",
                unsafe_allow_html=True,
            )
        elif msg["role"] == "assistant":
            st.markdown(
                f'<div class="chat-bubble-assistant">'
                f'<div class="role-label">⚡ Groq Assistant</div>{msg["content"]}'
                f"</div>",
                unsafe_allow_html=True,
            )

# ── Input form ────────────────────────────────────────────────────────────────
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Message",
        placeholder="Ask me anything… (Press Enter or click Send)",
        label_visibility="collapsed",
    )
    send_col, _ = st.columns([1, 5])
    with send_col:
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

# ── Handle submission ─────────────────────────────────────────────────────────
if submitted and user_input.strip():
    # Update system message if changed in sidebar
    st.session_state.messages[0] = build_system_message(system_prompt)

    # Append user message
    st.session_state.messages.append(build_user_message(user_input.strip()))

    # Display user bubble immediately
    st.markdown(
        f'<div class="chat-bubble-user">'
        f'<div class="role-label">You</div>{user_input.strip()}'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Stream assistant response
    st.markdown(
        '<div class="chat-bubble-assistant"><div class="role-label">⚡ Groq Assistant</div>',
        unsafe_allow_html=True,
    )
    response_placeholder = st.empty()
    full_response = ""

    try:
        for chunk in stream_chat(
            messages=st.session_state.messages,
            model=model_choice,
            temperature=temperature,
        ):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

        # Save assistant reply
        st.session_state.messages.append(build_assistant_message(full_response))

        # Update stats (rough token estimate: ~4 chars per token)
        st.session_state.turn_count += 1
        st.session_state.token_count += len(full_response) // 4

    except EnvironmentError as e:
        st.error(f"❌ Configuration error: {e}")
    except Exception as e:
        st.error(f"❌ API error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
    st.rerun()
