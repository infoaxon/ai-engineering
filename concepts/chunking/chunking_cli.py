# chunking_cli.py

def sliding_window_chunking(text: str, chunk_size: int, overlap: int):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


# Large text
large_document = """
Insurance is a contract between an insurer and a policyholder. It provides financial protection against losses.
The policyholder pays a premium, and in return, the insurer covers certain risks.
Health insurance covers medical expenses, while life insurance provides a lump sum to beneficiaries upon the policyholder's death.
Motor insurance protects against damages and third-party liabilities in accidents.
Travel insurance covers unexpected events like trip cancellations, medical emergencies, or lost baggage.
Home insurance protects your property against fire, theft, and natural disasters.
Commercial insurance provides coverage for businesses, safeguarding against operational risks.
""" * 5


def main():
    while True:
        # Take user input
        try:
            chunk_size = int(input("Enter chunk size (words): "))
            overlap = int(input("Enter overlap (words): "))
        except ValueError:
            print("Please enter valid numbers.")
            continue

        chunks = sliding_window_chunking(large_document, chunk_size, overlap)

        print(f"\n=== Generated {len(chunks)} Chunks ===")
        for idx, c in enumerate(chunks[:5], 1):  # Show only first 5 chunks
            print(f"\n[Chunk {idx}]:\n{c}\n{'-'*40}")

        choice = input(
            "\nDo you want to try different values? (y/n): ").lower()
        if choice != 'y':
            break


if __name__ == "__main__":
    main()
