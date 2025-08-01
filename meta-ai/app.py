import streamlit as st
import requests
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Load embeddings and vectorstore
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

# Streamlit setup
st.set_page_config(page_title="Insurance Chat Assistant", layout="wide")
st.title("💼 Insurance Chat Assistant")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Query Ollama


def query_ollama(prompt, model="llama3.2"):
    try:
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False
        })
        res.raise_for_status()
        data = res.json()
        return data.get("response", "[Error: Unexpected response format]")
    except Exception as e:
        return f"[Error contacting Ollama API: {e}]"


# Chat input
with st.form("chat_form"):
    query = st.text_input(
        "Ask a question about your insurance documents:", key="user_input")
    submitted = st.form_submit_button("Send")

if submitted and query:
    # Vector search
    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build prompt
    prompt = f"""
    You are an intelligent assistant trained on insurance documents.
    Use the context below to answer the user's question.
    If the context does not contain the answer, say you don't know.

    Context:
    {context}

    Question:
    {query}
    """

    # Get answer
    answer = query_ollama(prompt)
    st.session_state.chat_history.append((query, answer, docs))

# Display chat history
for user_msg, response, sources in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(response)
        with st.expander("📄 Source Context"):
            for doc in sources:
                st.markdown(
                    f"**{doc.metadata.get('source', 'Unknown')}**\n\n{doc.page_content[:500]}...")
