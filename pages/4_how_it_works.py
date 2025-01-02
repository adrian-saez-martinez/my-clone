import streamlit as st
from graphviz import Digraph

# Title and Description
st.title("How It works")
st.markdown(
    """
    The AI assistant was designed to provide accurate and meaningful answers 
    to your questions about Adrián, by using a RAG system implemented using Langchain and Chroma. 
    Below, you’ll find an explanation of the entire process.
    """
)

# Textual Explanation
st.markdown(
    """
    ### Workflow explanation
    - A **user** asks a **question**.
    - The **question** is embedded and processed by the **retriever**, which searches for the most relevant **documents** in the Chroma database.
    - The **retriever** retrieves the **documents**, which are then **integrated** into a **prompt** along with the initial **question**.
    - This **prompt** is sent to the **LLM**, which generates an **answer** based on the **retrieved documents** and the initial **question**.
    - Finally, the **LLM** returns the **answer** to the **user**.
    """
)

# Diagram Representation
st.subheader("Visual representation")
st.write("The diagram below illustrates how the app processes a query and retrieves the answer:")

# Create a directed graph
dot = Digraph(format="png")

# Adjust direction and spacing
dot.attr(rankdir="LR", nodesep="0.6", ranksep="0.4", fontsize="10")

# Define nodes
dot.node("User", "User\n👤", shape="ellipse", style="filled", color="lightblue", fontsize="10")
dot.node("Question", "Question", shape="box", style="filled", color="lightcoral", fontsize="10")
dot.node("Retriever", "Retriever", shape="ellipse", style="filled", color="skyblue", fontsize="10")
dot.node("Documents", "Similar\ndocuments\n📄📄📄", shape="box", style="filled", color="yellow", fontsize="10")
dot.node("Prompt", "Prompt", shape="box", style="filled", color="orange", fontsize="10")
dot.node("LLM", "LLM", shape="ellipse", style="filled", color="purple", fontcolor="white", fontsize="10")

# Define edges
dot.edge("User", "Question", label="asks", color="black", fontsize="10")
dot.edge("Question", "Retriever", label="embedded", color="black", fontsize="10")
dot.edge("Retriever", "Documents", label="retrieves", color="black", fontsize="10")
dot.edge("Documents", "Prompt", label="integrated", color="black", fontsize="10")
dot.edge("Question", "Prompt", label="integrated", color="black", fontsize="10")
dot.edge("Prompt", "LLM", label="queries", color="black", fontsize="10")
dot.edge("LLM", "User", label="answers", color="black", fontsize="10")

# Render graph
st.graphviz_chart(dot)
