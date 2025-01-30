from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
import streamlit as st

# Constants for testing
LLM_MODEL_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = st.secrets["HF_TOKEN"]
DB_PATH = "./chroma_databases/allinfo_db"  # Unified Chroma database path
DEBUG = True

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
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=3)
        if similarity_score_threshold is not None:
            docs_and_scores = [
                (doc, score) for doc, score in docs_and_scores if score <= similarity_score_threshold
            ]
        return docs_and_scores
    except Exception as e:
        print(f"[ERROR] Document retrieval failed: {e}")
        return []

def generate_answer(docs_and_scores, query):
    """
    Generate an answer based on retrieved documents, their similarity scores, the user's query, and custom prompt instructions.
    """
    try:
        context = "\n\n".join([doc.page_content for doc, _ in docs_and_scores])
        if not context:
            context = "No relevant information found."

        system_message = """
            You are Adrián's personal assistant. Your role is to answer questions about Adrián based on the provided context.
            - Always respond in the third person, referring to "Adrián", "he", or "him."
            - Base your response only on the given context.
        """
        user_message = "Question: {query}\n\nContext:\n{context}"

        system_prompt = SystemMessagePromptTemplate.from_template(system_message)
        user_prompt = HumanMessagePromptTemplate.from_template(user_message)
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        llm = HuggingFaceEndpoint(repo_id=LLM_MODEL_REPO_ID, huggingfacehub_api_token=HF_TOKEN)
        chain = LLMChain(llm=llm, prompt=chat_prompt)

        inputs = {"query": query, "context": context}
        return chain.run(inputs)
    except Exception as e:
        print(f"[ERROR] Answer generation failed: {e}")
        return "An error occurred during answer generation."

if __name__ == "__main__":
    query = "What is the role of Adrian at CTCON?"
    print("\n[DEBUG] Retrieving documents...")
    docs_and_scores = retrieve_documents(query, similarity_score_threshold=0.1)

    if DEBUG:
        print("[DEBUG] Retrieved Documents and Scores:")
        for doc, score in docs_and_scores:
            print(f"Score: {score:.4f}\nContent: {doc.page_content}\nMetadata: {doc.metadata}\n")

    print("\n[DEBUG] Generating answer...")
    answer = generate_answer(docs_and_scores, query)
    print("\n[DEBUG] Final Answer:")
    print(answer)
