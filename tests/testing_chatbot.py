from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain import hub
from langchain.agents import AgentExecutor, load_tools
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools import Tool
from langchain.tools.render import render_text_description
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import streamlit as st

# Hugging Face API Configuration
LLM_MODEL_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_MODEL_REPO_ID_AGENT = "HuggingFaceH4/zephyr-7b-beta"
HF_TOKEN = st.secrets["HF_TOKEN"]

# RetrievalQA Tool Definition
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

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


def retrieval_tool_func(input_text):
    """
    Tool function to retrieve data from the resume database.
    """
    print("input_text:")
    print(input_text)
    db_path = "./chroma_databases/resume_db/"  # Define the path to your database
    retrieval_chain = get_retrieval_chain(db_path)
    result = retrieval_chain.invoke(input_text)

    # Format the response and source documents
    response = f"Answer:\n{result.get('result')}\n\n"
    response += "Source Documents:\n"
    for doc in result.get("source_documents", []):
        response += f"- {doc.page_content}\n"
    return response

retrieval_tool = Tool(
    name="Resume Retrieval Tool",
    func=retrieval_tool_func,
    description="Good for retrieving information about proffesional experience."
)

# Define Tools
tools = [retrieval_tool]

# Set up ReAct-style Prompt
prompt = hub.pull("hwchase17/react-json")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# Define the Agent
chat_model_with_stop = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id=LLM_MODEL_REPO_ID_AGENT,
        huggingfacehub_api_token=HF_TOKEN
    )
).bind(stop=["\nObservation"])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    | prompt
    | chat_model_with_stop
    | ReActJsonSingleInputOutputParser()
)

# Create the Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# Test the Agent
if __name__ == "__main__":
    query = "Tell me about your role at CTCON."
    response = agent_executor.invoke({"input": query})

    print("\nAgent Response:")
    print(response)
