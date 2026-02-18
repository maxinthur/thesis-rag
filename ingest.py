"""Ingest a PDF thesis into the vector store.

Usage:
    python ingest.py --pdf data/thesis.pdf
    python ingest.py --pdf data/thesis.pdf --chunk-size 1000 --overlap 200
"""

import argparse
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_pdf(pdf_path: str) -> list[dict]:
    """Load a PDF and return a list of {page, text} dicts."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    docs = []
    for page in pages:
        text = page.page_content.strip()
        if len(text) > 50:  # skip near-empty pages (covers, blanks)
            docs.append({
                "text": text,
                "page": page.metadata.get("page", 0) + 1,
                "source": Path(pdf_path).name,
            })
    print(f"  Loaded {len(docs)} pages from {pdf_path}")
    return docs


def chunk_documents(
    docs: list[dict],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[dict]:
    """Split documents into overlapping chunks, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["text"])
        for i, text in enumerate(splits):
            chunks.append({
                "text": text,
                "page": doc["page"],
                "source": doc["source"],
                "chunk_index": i,
            })

    print(f"  Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def store_chunks(chunks: list[dict], reset: bool = False) -> None:
    """Embed chunks and store in ChromaDB."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )

    if reset:
        try:
            client.delete_collection("thesis")
        except ValueError:
            pass

    collection = client.get_or_create_collection(
        name="thesis",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch insert (ChromaDB handles batching internally)
    ids = [f"chunk_{i:04d}" for i in range(len(chunks))]
    texts = [c["text"] for c in chunks]
    metadatas = [{"page": c["page"], "source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]

    collection.add(documents=texts, ids=ids, metadatas=metadatas)
    print(f"  Stored {len(chunks)} chunks in ChromaDB (HNSW, cosine)")


def ingest(pdf_path: str, chunk_size: int, chunk_overlap: int, reset: bool = True) -> int:
    """Full ingestion pipeline. Returns number of chunks created."""
    print(f"\n{'='*60}")
    print("INGESTION PIPELINE")
    print(f"{'='*60}")

    t0 = time.time()

    print("\n[1/3] Loading PDF...")
    docs = load_pdf(pdf_path)

    print("\n[2/3] Chunking...")
    chunks = chunk_documents(docs, chunk_size, chunk_overlap)

    print("\n[3/3] Embedding & storing...")
    store_chunks(chunks, reset=reset)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — {len(chunks)} chunks indexed.\n")
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF into the thesis RAG system.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the thesis PDF")
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(f"Error: {args.pdf} not found.")
        sys.exit(1)

    ingest(args.pdf, args.chunk_size, args.overlap)
