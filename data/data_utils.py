from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings  # Updated import
import streamlit as st

def process_and_upload_text(file_path, db_path, topic_tag=None):
    """
    Process a text file and upload it to the Chroma database.

    Args:
        file_path (str): Path to the text file.
        db_path (str): Path to the Chroma database.
        topic_tag (str, optional): Tag or context to prepend to the text.

    Returns:
        None
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Split text into smaller chunks
    chunks = text.split("\n\n")  # Split by paragraphs

    # Prepend topic tag to each chunk
    if topic_tag:
        chunks = [f"{topic_tag}:{chunk}" for chunk in chunks]

    # Convert chunks into documents
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Embed and upload to Chroma
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # Switched to OpenAI embeddings
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )    
    vectorstore.add_documents(documents)
    print(f"Data from {file_path} uploaded successfully to {db_path}")
