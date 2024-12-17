from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st 

# Environment Variables
LLM_MODEL_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = st.secrets["HF_TOKEN"]

# Initialize Retrieval Chain
def get_retrieval_chain(db_path):
    """
    Create and return a RetrievalQA chain using Chroma and HuggingFace embeddings.

    Args:
        db_path (str): Path to the Chroma vectorstore directory.

    Returns:
        RetrievalQA: The retrieval-based question-answering chain.
    """
    # Initialize embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

    # Set up retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # Retrieve top 5 matches

    # Initialize Hugging Face model via HuggingFaceEndpoint
    llm = HuggingFaceEndpoint(
        repo_id=LLM_MODEL_REPO_ID,
        model_kwargs={
            "max_length": 2048,
        },
        huggingfacehub_api_token=HF_TOKEN
    )

    # Create RetrievalQA chain
    retrieval_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return retrieval_chain

# Query the Retrieval Chain
def retrieve_resume_data(query, db_path):
    """
    Query the Chroma vectorstore to retrieve professional experience information.

    Args:
        query (str): The natural language query.
        db_path (str): Path to the Chroma database.

    Returns:
        None
    """
    retrieval_chain = get_retrieval_chain(db_path)
    result = retrieval_chain.invoke(query)


    # Display answer and source documents
    print("\nAnswer:\n", result.get("result"))
    print("\nSource Documents:")
    for doc in result.get("source_documents", []):
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)

# Example Usage
if __name__ == "__main__":
    db_path = "./db/professional_experience_db/"  # Path to your Chroma database
    query = "Tell me about your role at CTCON."
    retrieve_resume_data(query, db_path)
