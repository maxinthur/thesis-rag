"""Retrieve relevant context and generate answers using Claude."""

import anthropic
import sys
from unittest.mock import MagicMock
sys.modules.setdefault('opentelemetry.exporter.otlp.proto.grpc.trace_exporter', MagicMock())
sys.modules.setdefault('opentelemetry.exporter.otlp.proto.grpc.exporter', MagicMock())
import chromadb
import httpx

import config


def get_collection():
    """Get the ChromaDB thesis collection."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_collection(name="thesis")


def embed_query(query: str) -> list[float]:
    """Embed a query using Voyage AI REST API."""
    resp = httpx.post(
        "https://api.voyageai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {config.VOYAGE_API_KEY}"},
        json={"input": [query], "model": config.EMBEDDING_MODEL, "input_type": "query"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def retrieve(query: str, top_k: int = config.TOP_K) -> list[dict]:
    """Retrieve the top-k most relevant chunks for a query.

    Returns a list of dicts with keys: text, page, score, rank.
    """
    collection = get_collection()
    query_embedding = embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "page": results["metadatas"][0][i].get("page", "?"),
            "score": 1 - results["distances"][0][i],  # cosine similarity
            "rank": i + 1,
        })
    return chunks


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context string."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] (page {chunk['page']}, similarity={chunk['score']:.3f})\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, context: str) -> str:
    """Generate an answer using Claude, grounded in the provided context."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    user_message = f"""Context passages from the thesis:

{context}

---

Question: {query}"""

    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        system=config.SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def ask(query: str, top_k: int = config.TOP_K) -> dict:
    """Full RAG pipeline: retrieve context, generate answer.

    Returns dict with: answer, chunks, query.
    """
    chunks = retrieve(query, top_k=top_k)
    context = build_context(chunks)
    answer = generate_answer(query, context)

    return {
        "query": query,
        "answer": answer,
        "chunks": chunks,
    }