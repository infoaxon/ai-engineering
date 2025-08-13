from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import requests

# Load embedding and DB
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)


def query_ollama(prompt, model="llama3.2"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
    )
    return response.json()["response"]


# Interactive loop
while True:
    query = input("Ask your insurance question (or 'exit'): ")
    if query.lower() == "exit":
        break

    docs = db.similarity_search(query, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    full_prompt = f"""You are an insurance assistant.
Answer the question using the context below. If it's not in the context, say "I don't know."

Context:
{context}

Question:
{query}
"""
    answer = query_ollama(full_prompt)
    print(f"\n Answer:\n{answer}\n")
