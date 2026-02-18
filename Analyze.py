"""Analyze PDF and chunking statistics.

Usage:
    python analyze.py --pdf data/thesis.pdf
    python analyze.py --pdf data/thesis.pdf --chunk-sizes 500 800 1000 1500 2000
"""

import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def analyze_pdf(pdf_path: str, chunk_sizes: list[int] = None):
    if chunk_sizes is None:
        chunk_sizes = [500, 800, 1000, 1500, 2000]

    # --- Load PDF ---
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    pages = [p for p in pages if len(p.page_content.strip()) > 50]

    page_lengths = [len(p.page_content) for p in pages]
    full_text = "\n\n".join(p.page_content for p in pages)

    print(f"\n{'='*60}")
    print(f"PDF ANALYSIS: {Path(pdf_path).name}")
    print(f"{'='*60}")

    print(f"\n--- Document Level ---")
    print(f"  Total pages (non-empty):  {len(pages)}")
    print(f"  Total characters:         {len(full_text):,}")
    print(f"  Total words (approx):     {len(full_text.split()):,}")
    print(f"  Avg chars/page:           {sum(page_lengths) / len(page_lengths):,.0f}")
    print(f"  Min chars/page:           {min(page_lengths):,}")
    print(f"  Max chars/page:           {max(page_lengths):,}")

    # --- Page length distribution ---
    print(f"\n--- Page Length Distribution ---")
    buckets = [0, 500, 1000, 1500, 2000, 2500, 3000, 5000, 10000]
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        count = sum(1 for l in page_lengths if lo <= l < hi)
        bar = "█" * count
        if count > 0:
            print(f"  {lo:>5}–{hi:<5} chars: {bar} ({count})")

    # --- Paragraph analysis ---
    paragraphs = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 20]
    para_lengths = [len(p) for p in paragraphs]
    print(f"\n--- Paragraph Level ---")
    print(f"  Total paragraphs:         {len(paragraphs)}")
    print(f"  Avg chars/paragraph:      {sum(para_lengths) / len(para_lengths):,.0f}")
    print(f"  Median chars/paragraph:   {sorted(para_lengths)[len(para_lengths)//2]:,}")
    print(f"  90th percentile:          {sorted(para_lengths)[int(len(para_lengths)*0.9)]:,}")
    print(f"  Max chars/paragraph:      {max(para_lengths):,}")

    # --- Chunking comparison ---
    print(f"\n--- Chunking Comparison ---")
    print(f"  {'chunk_size':>10} | {'overlap':>7} | {'n_chunks':>8} | {'avg_len':>7} | {'min_len':>7} | {'max_len':>7}")
    print(f"  {'-'*10}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

    for cs in chunk_sizes:
        overlap = max(50, cs // 4)  # 25% overlap
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cs,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(full_text)
        lengths = [len(c) for c in chunks]
        print(
            f"  {cs:>10} | {overlap:>7} | {len(chunks):>8} | "
            f"{sum(lengths)/len(lengths):>7.0f} | {min(lengths):>7} | {max(lengths):>7}"
        )

    # --- Recommendation ---
    median_para = sorted(para_lengths)[len(para_lengths) // 2]
    p90_para = sorted(para_lengths)[int(len(para_lengths) * 0.9)]
    print(f"\n--- Recommendation ---")
    print(f"  Your median paragraph is {median_para} chars, 90th percentile is {p90_para} chars.")
    print(f"  chunk_size should be >= 90th percentile to keep most paragraphs intact.")
    print(f"  Suggested: chunk_size={round(p90_para / 100) * 100}, overlap={round(p90_para / 100) * 25}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=None)
    args = parser.parse_args()
    analyze_pdf(args.pdf, args.chunk_sizes)