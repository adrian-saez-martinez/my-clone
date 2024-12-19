from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st

# Environment Variables
LLM_MODEL_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = st.secrets["HF_TOKEN"]

# Initialize Retrieval Chain
def get_allinfo_retrieval_chain(db_path):
    """
    Create and return a RetrievalQA chain using the unified Chroma database.

    Args:
        db_path (str): Path to the Chroma vectorstore directory.

    Returns:
        RetrievalQA: The retrieval-based question-answering chain.
    """
    # Initialize embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

    # Set up retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 matches

    # Initialize Hugging Face model via HuggingFaceEndpoint
    llm = HuggingFaceEndpoint(
        repo_id=LLM_MODEL_REPO_ID,
        huggingfacehub_api_token=HF_TOKEN
    )

    # Create RetrievalQA chain
    retrieval_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return retrieval_chain

# Query the Unified Database
def retrieve_allinfo(query, db_path):
    """
    Query the unified Chroma database for relevant information.

    Args:
        query (str): The natural language query.
        db_path (str): Path to the unified Chroma database.

    Returns:
        None
    """
    print(f"Query: {query}")
    retrieval_chain = get_allinfo_retrieval_chain(db_path)
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
    db_path = "./chroma_databases/allinfo_db"
    query = "What companies where you working researching?"
    retrieve_allinfo(query, db_path)
