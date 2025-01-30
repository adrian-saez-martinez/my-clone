from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st
import os

LLM_MODEL_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = st.secrets["HF_TOKEN"]

# Initialize the LLM
def initialize_llm():
    return HuggingFaceEndpoint(
        repo_id=LLM_MODEL_REPO_ID,
        model_kwargs={"max_length": 2048},
        huggingfacehub_api_token=HF_TOKEN,
    )


# Initialize Retrieval Chain
def get_retrieval_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_databases/allinfo_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = initialize_llm()

    return RetrievalQA.from_llm(
        llm=llm, retriever=retriever, return_source_documents=True
    )


# Define the chatbot memory workflow
def create_chatbot_workflow():
    llm = initialize_llm()
    retrieval_chain = get_retrieval_chain()
    workflow = StateGraph(state_schema=MessagesState)

    # Define the model call function
    def call_model(state: MessagesState):
        # Define the system prompt
        system_prompt = """
            You are Adrián's personal assistant. Your role is to answer questions about Adrián.

            - You will be provided with relevant context to generate accurate answers. Use it as your unique source of information.
            - Directly address the question using the given context.
            - Always respond in the third person, referring to "Adrián", "he" or "him."
            - Well-structured and easy to read.
            - Always respond in a conversational and professional tone, narrating Adrián experiences or thoughts.
            - Do not include "Assistant:", "Response:", "Correct Answer:", "Explanation:" or similar sections in your response.
            - Focus on giving precise, engaging, and human-like answers to ensure the response feels natural and directly addresses the question.
        """
        # Add system prompt and previous messages
        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        # Check the latest user query
        latest_message = state["messages"][-1]
        if isinstance(latest_message, HumanMessage):
            user_query = latest_message.content
            retrieval_result = retrieval_chain(user_query)
            retrieved_answer = retrieval_result.get("result", "No encontré información relevante.")
            return {"messages": [AIMessage(content=retrieved_answer)]}

        # Default LLM response
        response = llm.invoke(messages)
        return {"messages": response}

    # Define the node and edge in the workflow
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    # Add memory persistence
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# Debugging retrieval results
def debug_retrieval(query):
    retrieval_chain = get_retrieval_chain()
    result = retrieval_chain(query)
    print("Retrieved Documents and Scores:")
    for doc, score in result.get("source_documents", []):
        print(f"Score: {score}, Content: {doc.page_content}")

# Example usage
#debug_retrieval("What is Adrián's educational background?")