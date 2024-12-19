import sys
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA

# Constants for environment and database
LLM_MODEL_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = st.secrets["HF_TOKEN"]
DB_PATH = "./chroma_databases/allinfo_db"  # Unified Chroma database path

# Define retrieval function
def retrieve_report(query):
    """
    Use RetrievalQA to retrieve reports based on a query.

    Args:
        query (str): The user's query.

    Returns:
        dict: Retrieval result with answer and source documents.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = HuggingFaceEndpoint(
        repo_id=LLM_MODEL_REPO_ID,
        model_kwargs={"max_length": 2048},
        huggingfacehub_api_token=HF_TOKEN,
    )

    retrieval_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return retrieval_chain.invoke(query)

# Streamlit Interface
st.subheader("Ask me anything about Adrián")

# User input for query
query = st.text_input("Type your question:")

if st.button("Search"):
    if query:
        # Retrieve reports using the chain
        with st.spinner("Retrieving information..."):
            try:
                result = retrieve_report(query)

                # Display the answer
                st.write("### Answer:")
                st.write(result.get("result"))
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
