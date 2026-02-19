"""Thesis RAG — Interactive Q&A over a PhD thesis.

A Retrieval-Augmented Generation system that answers questions about
'Reconstructing Neural Dynamics Underlying Cognitive Flexibility'
by Max Thurm (Heidelberg University, 2025).

Run locally:  streamlit run app.py
"""

import os
import time

import streamlit as st

import config
from retrieval import ask, retrieve, build_context

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Thesis RAG — Neural Dynamics",
    page_icon="🧠",
    layout="wide",
)

# ── Authentication ───────────────────────────────────────────
APP_PASSWORD = os.getenv("APP_PASSWORD", "thesisRAG2025!")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔒 Thesis RAG")
    st.caption("Enter the access code to continue.")
    pw = st.text_input("Access code:", type="password")
    if pw:
        if pw == APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect access code.")
    st.stop()

# ── Rate limiting ────────────────────────────────────────────
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_HOUR", "20"))

if "query_timestamps" not in st.session_state:
    st.session_state["query_timestamps"] = []


def check_rate_limit() -> bool:
    """Returns True if under the rate limit."""
    now = time.time()
    hour_ago = now - 3600
    st.session_state["query_timestamps"] = [
        t for t in st.session_state["query_timestamps"] if t > hour_ago
    ]
    return len(st.session_state["query_timestamps"]) < RATE_LIMIT


def record_query():
    st.session_state["query_timestamps"].append(time.time())


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.title("Thesis RAG")
    st.caption(
        "Retrieval-Augmented Generation over a PhD thesis on "
        "reconstructing neural dynamics underlying cognitive flexibility."
    )

    st.divider()

    top_k = st.slider("Retrieved chunks (top-k)", min_value=1, max_value=10, value=config.TOP_K)
    show_context = st.toggle("Show retrieved context", value=False)

    st.divider()

    st.markdown("**Example questions:**")
    examples = [
        "What is the pePLRNN?",
        "How does the brain implement rule learning?",
        "What is the mirage reconstruction problem?",
        "How was the model validated on non-stationary data?",
        "Do neural transitions happen before or after behavioral changes?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["user_query"] = ex

    st.divider()
    st.markdown(
        "Built by [Max Thurm](https://github.com/maxthurm) · "
        "Powered by Claude (Anthropic)"
    )
    queries_used = len([t for t in st.session_state.get("query_timestamps", []) if t > time.time() - 3600])
    st.caption(f"Queries: {queries_used}/{RATE_LIMIT} this hour")

# ── Main area ────────────────────────────────────────────────
st.header("Ask a question about the thesis")

query = st.text_input(
    "Your question:",
    value=st.session_state.get("user_query", ""),
    placeholder="e.g. What is the pePLRNN and what problem does it solve?",
    label_visibility="collapsed",
)

if query:
    if not check_rate_limit():
        remaining = 3600 - (time.time() - st.session_state["query_timestamps"][0])
        st.warning(f"Rate limit reached ({RATE_LIMIT} queries/hour). Try again in {int(remaining // 60)} minutes.")
    else:
        record_query()
        with st.spinner("Retrieving context and generating answer..."):
            result = ask(query, top_k=top_k)

        # ── Answer ───────────────────────────────────────────────
        st.markdown("### Answer")
        st.markdown(result["answer"])

        # ── Context (collapsible) ────────────────────────────────
        if show_context:
            st.markdown("### Retrieved Context")
            for i, chunk in enumerate(result["chunks"], 1):
                with st.expander(
                    f"Chunk {i} — page {chunk['page']} (similarity: {chunk['score']:.3f})"
                ):
                    st.markdown(chunk["text"])

        # ── Retrieval diagnostics ────────────────────────────────
        with st.expander("Retrieval diagnostics"):
            cols = st.columns(len(result["chunks"]))
            for i, (col, chunk) in enumerate(zip(cols, result["chunks"])):
                with col:
                    st.metric(f"Chunk {i+1}", f"{chunk['score']:.3f}", f"p.{chunk['page']}")