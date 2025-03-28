from langchain.embeddings import SentenceTransformerEmbeddings

embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
print(embedding.embed_query("test"))  # should return a non-empty vector

