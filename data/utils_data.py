from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import json
import streamlit as st

HF_TOKEN = st.secrets["HF_TOKEN"]

def upload_resume(json_path, db_path):
    """Upload resume data from JSON into Chroma."""
    # Load resume data from JSON
    with open(json_path, "r", encoding="utf-8") as f:
        resume_data = json.load(f)

    # Prepare documents
    documents = [
        Document(
            page_content=item["description"],
            metadata={key: value for key, value in item.items() if key != "description"}
        )
        for item in resume_data
    ]

    # Initialize Chroma and upload
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    vectorstore.add_documents(documents)
    print("Resume data uploaded successfully!")


