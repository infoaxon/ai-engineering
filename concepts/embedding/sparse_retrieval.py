#!/usr/bin/env python3
import sys
import numpy as np
from rank_bm25 import BM25Okapi


# --- 1. Load chunks from a file ---
def load_chunks(file_path):
    """
    Read chunks from a text file, one chunk per line.
    Empty lines are skipped.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# --- 2. Build BM25 index ---
def build_bm25(chunks):
    """
    Tokenize each chunk and build a BM25Okapi index.
    """
    tokenized = [c.split() for c in chunks]
    return BM25Okapi(tokenized)


# --- 3. Query loop ---
def query_loop(bm25, chunks, top_k=5):
    """
    Repeatedly prompt for a query, run BM25, and display top_k chunks.
    """
    print("\nEnter your query (blank to exit):")
    while True:
        query = input("> ").strip()
        if not query:
            print("Goodbye!")
            break

        # Get BM25 scores and rank
        scores = bm25.get_scores(query.split())
        idxs = np.argsort(scores)[::-1][:top_k]

        print(f'\nTop {top_k} results for "{query}":\n')
        for rank, i in enumerate(idxs, 1):
            print(f"{rank}. [Score {scores[i]:.2f}] {chunks[i][:100]}…\n")
        print("-" * 60 + "\n")


def main():
    if len(sys.argv) != 2:
        print("Usage: python sparse_retrieval.py <chunks_txt_file>")
        sys.exit(1)

    chunks_file = sys.argv[1]
    print(f"\nLoading chunks from {chunks_file} …")
    chunks = load_chunks(chunks_file)
    print(f"Loaded {len(chunks)} chunks.\n")

    print("Building BM25 index …")
    bm25 = build_bm25(chunks)
    print("Index ready. You can now search.\n")

    query_loop(bm25, chunks)


if __name__ == "__main__":
    main()
