# pdf_chunking_cli.py

import sys
from pypdf import PdfReader

# --- Chunking Logic ---
def sliding_window_chunking(text: str, chunk_size: int, overlap: int):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# --- PDF Text Extraction ---
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# --- ASCII Visualization ---
def ascii_visualization(total_words, chunk_size, overlap):
    step = chunk_size - overlap
    pos = 0
    print("\nASCII Visualization of Chunking:")
    while pos < total_words:
        print(" " * (pos // 2) + "[" + "=" * (chunk_size // 2) + f"] ({pos})")
        pos += step

# --- Main CLI ---
def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_chunking_cli.py <pdf_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"\nReading PDF: {file_path}")
    text = extract_text_from_pdf(file_path)
    total_words = len(text.split())
    print(f"Extracted {total_words} words from the PDF.")

    while True:
        try:
            chunk_size = int(input("Enter chunk size (words): "))
            overlap = int(input("Enter overlap (words): "))
        except ValueError:
            print("Please enter valid numbers.")
            continue
        
        chunks = sliding_window_chunking(text, chunk_size, overlap)
        print(f"\n=== Generated {len(chunks)} Chunks ===\n")
        
        # Show first few chunks
        for idx, c in enumerate(chunks[:5], 1):  # Show only first 5 chunks
            print(f"[Chunk {idx}]:\n{c}\n{'-'*50}")

        # ASCII visualization
        ascii_visualization(total_words, chunk_size, overlap)

        choice = input("\nDo you want to try different values? (y/n): ").lower()
        if choice != 'y':
            break

if __name__ == "__main__":
    main()

