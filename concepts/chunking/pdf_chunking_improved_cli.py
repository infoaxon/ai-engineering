#!/usr/bin/env python3

import sys
from pypdf import PdfReader


# --- Chunking Logic ---
def sliding_window_chunking(text: str, chunk_size: int, overlap: int):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i: i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


# --- PDF Text Extraction ---
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            # Preserve paragraph breaks
            text += page_text.strip() + "\n\n"
    return text.strip()


# --- Main CLI ---
def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_chunking_cli.py <pdf_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"\nReading PDF: {file_path}")
    text = extract_text_from_pdf(file_path)
    total_words = len(text.split())
    print(f"Extracted {total_words} words from the PDF.\n")

    while True:
        try:
            chunk_size = int(input("Enter chunk size (words): "))
            overlap = int(input("Enter overlap (words): "))
            if overlap >= chunk_size:
                print(
                    "Error: overlap must be less than chunk size. Please try again.\n"
                )
                continue
        except ValueError:
            print("Please enter valid integers for chunk size and overlap.\n")
            continue

        chunks = sliding_window_chunking(text, chunk_size, overlap)
        print(f"\n=== Generated {len(chunks)} Chunks ===\n")

        separator = "\n" + "=" * 80 + "\n"
        for idx, c in enumerate(chunks, 1):
            print(f"[Chunk {idx}]\n{c}{separator}")

        choice = input("Try different values? (y/n): ").strip().lower()
        if choice != "y":
            break

    print("\nDone. Each chunk is separated by a line of '=' for easy distinction.")


if __name__ == "__main__":
    main()
