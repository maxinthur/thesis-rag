"""Thesis RAG — Interactive Q&A over a PhD thesis.

A Retrieval-Augmented Generation system that answers questions about
'Reconstructing Neural Dynamics Underlying Cognitive Flexibility'
by Max Thurm (Heidelberg University, 2025).

Run locally:  streamlit run app.py
"""

import streamlit as st

import config
from retrieval import ask, retrieve, build_context

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Thesis RAG — Neural Dynamics",
    page_icon="🧠",
    layout="wide",
)

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

# ── Main area ────────────────────────────────────────────────
st.header("Ask a question about the thesis")

query = st.text_input(
    "Your question:",
    value=st.session_state.get("user_query", ""),
    placeholder="e.g. What is the pePLRNN and what problem does it solve?",
    label_visibility="collapsed",
)

if query:
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
