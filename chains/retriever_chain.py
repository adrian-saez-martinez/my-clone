from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st 
import os

# Constants for environment and database
LLM_MODEL_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = st.secrets["HF_TOKEN"]
DB_PATH = "./chroma_databases/allinfo_db"
DEBUG = False

# Define retrieval function
def retrieve_documents(query, similarity_score_threshold=None):
    """
    Retrieve documents relevant to the query along with their similarity scores.
    Optionally filter documents based on a similarity score threshold.

    Args:
        query (str): The user's query.
        similarity_score_threshold (float, optional): Minimum similarity score to include a document.

    Returns:
        list of tuples: Each tuple contains a Document and its similarity score.
    """
    
    # Initialize embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # Perform similarity search with scores
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=3)

    # Filter results by similarity score threshold if provided
    if similarity_score_threshold is not None:
        docs_and_scores = [
            (doc, score) for doc, score in docs_and_scores
            if score <= similarity_score_threshold  # Lower scores are more similar
        ]

    return docs_and_scores