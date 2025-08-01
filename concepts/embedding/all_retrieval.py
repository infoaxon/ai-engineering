#!/usr/bin/env python3
"""
rag_retrieval_cli.py

A unified CLI tool demonstrating:
  1) Sparse Retrieval (BM25)
  2) Dense Retrieval (Cosine Similarity on embeddings)
  3) Hybrid Retrieval (score fusion)

Usage:
  python rag_retrieval_cli.py <chunks_txt_file>

Requirements:
  pip install rank-bm25 sentence-transformers numpy scikit-learn
"""

import sys
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_chunks(file_path):
    """Read chunks from a text file (one chunk per line)."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_bm25_index(chunks):
    """Tokenize and build BM25 index."""
    tokenized = [c.split() for c in chunks]
    return BM25Okapi(tokenized)


def build_dense_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings for each chunk."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=False)
    return embeddings, model


def normalize_scores(arr):
    """Min-max normalize an array to [0,1]."""
    min_, max_ = arr.min(), arr.max()
    if max_ - min_ < 1e-9:
        return np.zeros_like(arr)
    return (arr - min_) / (max_ - min_)


def bm25_retrieve(bm25, chunks, query, top_k=5):
    scores = bm25.get_scores(query.split())
    idxs = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], scores[i]) for i in idxs], scores


def dense_retrieve(embeddings, chunks, model, query, top_k=5):
    q_emb = model.encode([query])
    sims = cosine_similarity(q_emb, embeddings)[0]
    idxs = np.argsort(sims)[::-1][:top_k]
    return [(chunks[i], sims[i]) for i in idxs], sims


def hybrid_retrieve(bm25_scores, dense_sims, chunks, top_k=5, alpha=0.5):
    bm25_n = normalize_scores(bm25_scores)
    dense_n = normalize_scores(dense_sims)
    combined = alpha * dense_n + (1 - alpha) * bm25_n
    idxs = np.argsort(combined)[::-1][:top_k]
    return [(chunks[i], combined[i]) for i in idxs]


def interactive_search(bm25, embeddings, model, chunks):
    print("\nEnter your query (blank to exit):")
    while True:
        query = input("> ").strip()
        if not query:
            print("Exiting.")
            break

        print("\n-- BM25 Results --")
        bm25_results, bm25_scores = bm25_retrieve(bm25, chunks, query)
        for text, score in bm25_results:
            print(f"[BM25] {score:.2f} - {text[:80]}…")

        print("\n-- Dense Results --")
        dense_results, dense_scores = dense_retrieve(
            embeddings, chunks, model, query)
        for text, score in dense_results:
            print(f"[Dense] {score:.4f} - {text[:80]}…")

        print("\n-- Hybrid Results (α=0.5) --")
        hybrid_results = hybrid_retrieve(
            bm25_scores, dense_scores, chunks, alpha=0.5)
        for text, score in hybrid_results:
            print(f"[Hybrid] {score:.4f} - {text[:80]}…")

        print("\n" + "=" * 60 + "\n")


def main():
    if len(sys.argv) != 2:
        print("Usage: python rag_retrieval_cli.py <chunks_txt_file>")
        sys.exit(1)

    chunks_file = sys.argv[1]
    print(f"\nLoading chunks from: {chunks_file}")
    chunks = load_chunks(chunks_file)
    print(f"Loaded {len(chunks)} chunks.\n")

    print("Building BM25 index…")
    bm25 = build_bm25_index(chunks)
    print("BM25 ready.\n")

    print("Generating dense embeddings…")
    embeddings, model = build_dense_embeddings(chunks)
    print(f"Embeddings shape: {embeddings.shape}\n")

    interactive_search(bm25, embeddings, model, chunks)


if __name__ == "__main__":
    main()
