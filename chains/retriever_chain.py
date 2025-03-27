from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants for environment and database
DB_PATH = "./chroma_databases/allinfo_db"
DEBUG = False

# Initialize embeddings and vectorstore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

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
    # Perform similarity search with scores
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=3)

    # Filter results by similarity score threshold if provided
    if similarity_score_threshold is not None:
        docs_and_scores = [
            (doc, score) for doc, score in docs_and_scores
            if score <= similarity_score_threshold  # Lower scores are more similar
        ]

    return docs_and_scores

@tool(response_format="content_and_artifact")
def retriever(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vectorstore.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs