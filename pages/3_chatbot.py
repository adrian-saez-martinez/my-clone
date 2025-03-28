import sys
import os

# Check if running in Streamlit Cloud
if os.getenv("CLOUD_SERVER") == "true":
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

# Constants for environment and database
LLM_MODEL_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = st.secrets["HF_TOKEN"]
DB_PATH = "./chroma_databases/allinfo_db"  # Unified Chroma database path
DEBUG = True
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


# Define the LLM chain with a custom prompt
def generate_answer(docs_and_scores, query):
    """
    Generate an answer based on retrieved documents, their similarity scores, the user's query, and custom prompt instructions.

    Args:
        docs_and_scores (list): List of tuples (Document, similarity score).
        query (str): The user's query.

    Returns:
        str: Generated answer.
    """
    if docs_and_scores:
        # Prepare the context by including similarity scores
        debug_context = ""
        context = ""
        for doc, score in docs_and_scores:
            # Context for debugging/logging
            debug_context += f"Document (Similarity Score: {score:.4f}):\n{doc.page_content}\n\n"

            # Clean context for the LLM
            context += f"{doc.page_content}\n\n"
        # Define the custom prompt
        system_message = """
        You are Adrián's personal assistant. Your role is to answer questions about Adrián based on the provided context.

        - The context may contain tags such as:
        - "About my professional life:" for professional experiences.
        - "This is a general thought Adrián has:" for personal thoughts or opinions.
        - "This is about Adrián's personal life or free time:" for personal life content.

        - Use these tags to interpret the context correctly and generate a suitable response.
        - Always refer to Adrián in the third person, using "he" or "him."
        - If multiple tags appear, prioritize those that directly relate to the question.
        - Structure the response well and maintain a professional, conversational tone.
        - If no relevant context is found, acknowledge it honestly and provide a general response based on the question.
        """
        user_message = "Question: {query}\n\nContext:\n{context}"

    else:
        # Generic response when no documents are retrieved
        context = "No relevant information found."
        system_message = """
        You are Adrián's personal assistant. Your role is to answer questions about Adrián based on the provided context.

        - The context may contain tags such as:
        - "About my professional life:" for professional experiences.
        - "This is a general thought Adrián has:" for personal thoughts or opinions.
        - "This is about Adrián's personal life or free time:" for personal life content.

        - Use these tags to interpret the context correctly and generate a suitable response.
        - Always refer to Adrián in the third person, using "he" or "him."
        - If multiple tags appear, prioritize those that directly relate to the question.
        - Structure the response well and maintain a professional, conversational tone.
        - If no relevant context is found, acknowledge it honestly.
        """
        user_message = "Question: {query}\n\nContext:\n{context}"

    # Create prompt templates
    system_prompt = SystemMessagePromptTemplate.from_template(system_message)
    user_prompt = HumanMessagePromptTemplate.from_template(user_message)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

    # Initialize the Hugging Face LLM
    llm = HuggingFaceEndpoint(
        repo_id=LLM_MODEL_REPO_ID,
        huggingfacehub_api_token=HF_TOKEN,
    )

    # Create the LLM chain
    chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
    )

    # Generate the final answer
    inputs = {"query": query, "context": context}
    return chain.run(inputs)

# Streamlit Interface
st.markdown("### Ask me anything about Adrián")

# User input for query
query = st.text_input("Type your question:")

if st.button("Ask"):
    if query:
        # Retrieve documents and generate an answer
        with st.spinner("Retrieving information..."):
            try:
                docs_and_scores = retrieve_documents(query,0.9)

                # Generate the answer based on documents or a generic prompt
                answer = generate_answer(docs_and_scores, query)

                # Display the answer
                st.write("### Answer:")
                st.write(answer)
                if DEBUG:
                    # Display the related documents with similarity scores
                    if docs_and_scores:
                        st.write("### Related Documents:")
                        for doc, score in docs_and_scores:
                            st.write(f"**Similarity Score:** {score:.4f}")
                            st.write(f"**Content:** {doc.page_content}")
                            st.write(f"**Metadata:** {doc.metadata}")
                            st.write("---")
                    else:
                        st.warning("No relevant information found.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
