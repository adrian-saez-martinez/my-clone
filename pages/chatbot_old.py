import sys
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
DB_PATH = "./chroma_databases/allinfo_db"

# Define retrieval function
def retrieve_documents(query):
    """
    Retrieve documents relevant to the query.

    Args:
        query (str): The user's query.

    Returns:
        list: Retrieved documents.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    return retriever.get_relevant_documents(query)

# Define the LLM chain with a custom prompt
def generate_answer(documents, query):
    """
    Generate an answer based on retrieved documents, the user's query, and custom prompt instructions.

    Args:
        documents (list): Retrieved documents.
        query (str): The user's query.

    Returns:
        str: Generated answer.
    """
    # Check if documents are retrieved
    if documents:
        # Format retrieved documents into a readable string
        context = "\n\n".join([f"{doc.page_content}" for doc in documents])

        # Define the custom prompt
        system_message = """
                You are an AI assistant with extensive knowledge about Adrián's professional experience, personal interests, and thoughts.
                Use the provided context and the user's query to answer as accurately as possible. Be concise and provide relevant details.
                """
        user_message = "Question: {query}\n\nRetrieved Context:\n{context}"

    else:
        # Generic prompt if no documents are retrieved
        context = "No relevant information found about that."
        system_message = """
                You are an AI assistant answering questions about Adrián. Unfortunately, no relevant information was retrieved from the database.
                Answer the user's query as best as possible based on general context.
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
st.subheader("Ask me anything about Adrián")

# User input for query
query = st.text_input("Type your question:")

if st.button("Search"):
    if query:
        # Retrieve documents and generate an answer
        with st.spinner("Retrieving information..."):
            try:
                documents = retrieve_documents(query)

                # Generate the answer based on documents or a generic prompt
                answer = generate_answer(documents, query)

                # Display the answer
                st.write("### Answer:")
                st.write(answer)

                # Display the related documents if any
                if documents:
                    st.write("### Related Documents:")
                    for doc in documents:
                        st.write(f"**Content:** {doc.page_content}")
                        st.write(f"**Metadata:** {doc.metadata}")
                        st.write("---")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
