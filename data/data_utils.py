from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st

HF_TOKEN = st.secrets["HF_TOKEN"]

def process_and_upload_text(file_path, db_path):
    """Process text file and upload to Chroma database."""
    # Read text file
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split text into smaller chunks
    chunks = text.split("\n\n")  # Split by paragraphs

    # Convert chunks into documents
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Embed and upload to Chroma
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )    
    vectorstore.add_documents(documents)
    print(f"Data from {file_path} uploaded successfully to {db_path}")
