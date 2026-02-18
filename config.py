"""Central configuration for the Thesis RAG system."""

import os
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# --- Embedding Model (Voyage AI) ---
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
EMBEDDING_MODEL = "voyage-3-large"
EMBEDDING_DIM = 1024

# --- LLM ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS = 1024

# --- Chunking ---
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 700

# --- Retrieval ---
TOP_K = 6

# --- System Prompt ---
SYSTEM_PROMPT = """You are a research assistant that answers questions about a PhD thesis on 
reconstructing neural dynamics underlying cognitive flexibility using parameter-evolving RNNs.

Rules:
1. Answer ONLY based on the provided context passages.
2. If the context does not contain enough information, say so explicitly.
3. Cite which passage(s) you used by referencing their index [1], [2], etc.
4. Be precise and technical when the question demands it, accessible when it does not.
5. Never fabricate information that is not in the context."""