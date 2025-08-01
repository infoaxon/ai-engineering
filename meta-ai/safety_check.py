# Check embeddings
texts = [chunk.page_content for chunk in chunks]
sample_embedding = embedding.embed_query("insurance")  # sanity check

if not texts or not sample_embedding:
    raise Exception("No content or embeddings failed.")

# Create vector store
db = Chroma.from_documents(chunks, embedding, persist_directory="./chroma_db")
