from langchain.embeddings import SentenceTransformerEmbeddings

print("Loaded chunks:", len(chunks))
print("Sample chunk:", chunks[0].page_content[:200])
