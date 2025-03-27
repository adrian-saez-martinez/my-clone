from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from chains.retriever_chain import retriever, retrieve_documents
from langgraph.prebuilt import ToolNode, tools_condition

# Initialize the LLM
def initialize_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=2048,
    )

# Define the chatbot memory workflow
def create_chatbot_workflow():
    llm = initialize_llm()
    workflow = StateGraph(state_schema=MessagesState)

    # Step 1: Generate an AIMessage that may include a tool-call to be sent.
    def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools([retriever])
        response = llm_with_tools.invoke(state["messages"])
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}

    # Step 2: Execute the retrieval.
    tools = ToolNode([retriever])

    # Step 3: Generate a response using the retrieved content.
    def generate(state: MessagesState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )

        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = llm.invoke(prompt)
        return {"messages": [response]}

    # Define the node and edge in the workflow
    workflow.add_edge(START, "query_or_respond")
    workflow.add_node("query_or_respond", query_or_respond)
    workflow.add_node("tools", tools)
    workflow.add_node("generate", generate)
    workflow.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    workflow.add_edge("tools", "generate")
    workflow.add_edge("generate", END)

    # Add memory persistence
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

# Debugging retrieval results
def debug_retrieval(query):
    result = retrieve_documents(query)
    print("Retrieved Documents and Scores:")
    for doc, score in result:
        print(f"Score: {score}, Content: {doc.page_content}")