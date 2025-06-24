"""
Streamlit front-end for the AutoML-GPT chatbot.

Run with:
    streamlit run src/components/app.py
"""

from __future__ import annotations
import streamlit as st
import sys
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if ROOT not in map(pathlib.Path, sys.path):
    sys.path.append(str(ROOT))
from src.components.chatbot import ask

st.set_page_config(page_title="AutoML-GPT", page_icon="🤖", layout="wide")
st.title("AutoML-GPT - Interactive Chatbot")

# Keep conversation in session_state
if "history" not in st.session_state:
    st.session_state.history: list[tuple[str, str]] = [] # type: ignore

# Display previous turns
for role, text in st.session_state.history:
    avatar = "🧑‍💻" if role == "user" else "🤖"
    st.chat_message(avatar).markdown(text)

# Chat input box at the bottom
if user_input := st.chat_input("Ask about the project, models, or type 'exit' to quit…"):
    # Append user turn
    st.session_state.history.append(("user", user_input))

    with st.spinner("Thinking…"):
        bot_reply = ask(user_input)

    # Append bot turn
    st.session_state.history.append(("bot", bot_reply))

    # Rerender to show the new messages
    st.experimental_rerun()