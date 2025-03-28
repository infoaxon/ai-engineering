from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader, TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load all PDFs and text files
loader = DirectoryLoader("/Users/shubhamnagar/ai-engineering/meta-ai/mibl-training-docs/allpdfs", loader_cls=PyMuPDFLoader)
docs = loader.load()

# Check the document count and do inspection

print(f"Documents loaded: {len(docs)}")
print("Sample document:", docs[0].page_content[:300])



# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# Safety Check Cycle

if not chunks:
    raise ValueError("No chunks were created. Check if your documents are empty.")

texts = [chunk.page_content for chunk in chunks if chunk.page_content.strip()]
if not texts:
    raise ValueError("Chunks exist but they are empty.")

# Load Embedding Model

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Test Embedding Model

sample_vec = embedding.embed_query("insurance")
if not sample_vec:
    raise ValueError("Embedding model failed to return vector.")

# Embed chunks and save to Chroma
db = Chroma.from_documents(chunks, embedding, persist_directory="./chroma_db")
# db.persist()
print("Documents Embedded and stored")

