#!/usr/bin/env python3

import sys
import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --- 1. PDF Extraction ---
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text.strip() + "\n\n"
    return text.strip()


# --- 2. Chunking Logic ---
def sliding_window_chunking(text: str, chunk_size: int, overlap: int):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i: i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


# --- 3. Build BM25 Index ---
def build_bm25(chunks):
    tokenized = [c.split() for c in chunks]
    return BM25Okapi(tokenized)


# --- 4. Build Embeddings ---
def build_embeddings(chunks, model):
    return model.encode(chunks, show_progress_bar=False)


# --- 5. Retrieval Methods ---
def retrieve_bm25(bm25, chunks, query, top_k=3):
    scores = bm25.get_scores(query.split())
    idxs = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], scores[i]) for i in idxs]


def retrieve_vector(chunk_emb, chunks, query, model, top_k=3):
    q_emb = model.encode([query])
    sims = cosine_similarity(q_emb, chunk_emb)[0]
    idxs = np.argsort(sims)[::-1][:top_k]
    return [(chunks[i], sims[i]) for i in idxs]


def retrieve_hybrid(bm25, chunk_emb, chunks, query, model, alpha=0.5, top_k=3):
    # bm25
    bm25_scores = bm25.get_scores(query.split())
    # vector
    q_emb = model.encode([query])
    vec_scores = cosine_similarity(q_emb, chunk_emb)[0]
    # normalize
    bmin, bmax = bm25_scores.min(), bm25_scores.max()
    vmin, vmax = vec_scores.min(), vec_scores.max()
    bm25_n = (bm25_scores - bmin) / (bmax - bmin + 1e-9)
    vec_n = (vec_scores - vmin) / (vmax - vmin + 1e-9)
    comb = alpha * vec_n + (1 - alpha) * bm25_n
    idxs = np.argsort(comb)[::-1][:top_k]
    return [(chunks[i], comb[i]) for i in idxs]


# --- 6. CLI ---
def main():
    if len(sys.argv) < 2:
        print("Usage: python rag_cli.py <pdf_file>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"\n1) Extracting text from PDF → {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    total_w = len(text.split())
    print(f"   → Extracted {total_w} words.\n")

    # Chunking parameters
    while True:
        try:
            cs = int(input("Enter chunk size (words): "))
            ov = int(input("Enter overlap    (words): "))
            if ov >= cs:
                print("  ← overlap must be less than chunk size. Try again.\n")
                continue
        except ValueError:
            print("  ← please enter integers. Try again.\n")
            continue
        break

    # Chunk
    print("\n2) Chunking document…")
    chunks = sliding_window_chunking(text, cs, ov)
    print(f"   → Generated {len(chunks)} chunks.\n")
    # (Optionally show first few)
    for i, c in enumerate(chunks[:3], 1):
        print(f"[Chunk {i}]: {c[:80]}…\n")

    # Load model & embed
    print("3) Loading embedding model & generating embeddings…")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_emb = build_embeddings(chunks, model)
    print(f"   → Embeddings shape: {chunk_emb.shape}\n")

    # Build BM25
    print("4) Building BM25 index…")
    bm25 = build_bm25(chunks)
    print("   → Ready for retrieval.\n")

    # Interactive query
    while True:
        q = input("Enter query (or blank to exit): ").strip()
        if not q:
            break

        print("\n>>> BM25 Retrieval:")
        for txt, sc in retrieve_bm25(bm25, chunks, q):
            print(f"  • Score {sc:.2f} → {txt[:80]}…")

        print("\n>>> Vector Retrieval:")
        for txt, sc in retrieve_vector(chunk_emb, chunks, q, model):
            print(f"  • Score {sc:.4f} → {txt[:80]}…")

        print("\n>>> Hybrid Retrieval (α=0.5):")
        for txt, sc in retrieve_hybrid(bm25, chunk_emb, chunks, q, model):
            print(f"  • Score {sc:.4f} → {txt[:80]}…")

        print("\n" + "-" * 60 + "\n")

    print("Goodbye!")


if __name__ == "__main__":
    main()
