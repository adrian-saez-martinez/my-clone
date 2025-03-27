from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from chains.retriever_chain import retriever
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# Define a new custom state schema
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    original_language: str

# Initialize the LLM
def initialize_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=2048,
    )

# Define language detection and translation node
def detect_language_and_translate(state: State):
    llm = initialize_llm()
    user_message = state["messages"][-1].content

    detect_prompt = f"Identify the language of the following text. Respond with ONLY 'english' or 'spanish':\n\n{user_message}"
    detected_language = llm.invoke([HumanMessage(content=detect_prompt)]).content.strip().lower()

    if detected_language == "spanish":
        translate_prompt = f"Translate this from Spanish to English:\n\n{user_message}"
        translated_message = llm.invoke([HumanMessage(content=translate_prompt)]).content.strip()
        state["messages"][-1] = HumanMessage(content=translated_message)

    return {"messages": state["messages"], "original_language": detected_language}

# Generate an AIMessage with potential tool call
def query_or_respond(state: State):
    llm_with_tools = initialize_llm().bind_tools([retriever])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response], "original_language": state["original_language"]}

# Generate the final response using retrieved content
def generate(state: State):
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        f"\n\n{docs_content}"
    )

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = initialize_llm().invoke(prompt)
    return {"messages": [response], "original_language": state["original_language"]}

# Translate final response back to original language if needed
def translate_response_back(state: State):
    original_language = state["original_language"]
    response_message = state["messages"][-1].content

    if original_language == "spanish":
        translate_prompt = f"Translate this from English to Spanish:\n\n{response_message}"
        translated_response = initialize_llm().invoke([HumanMessage(content=translate_prompt)]).content.strip()
        state["messages"][-1] = AIMessage(content=translated_response)

    return {"messages": state["messages"]}

# Define and compile the workflow
def create_chatbot_workflow():
    workflow = StateGraph(state_schema=State)

    workflow.add_node("detect_language_and_translate", detect_language_and_translate)
    workflow.add_node("query_or_respond", query_or_respond)
    workflow.add_node("tools", ToolNode([retriever]))
    workflow.add_node("generate", generate)
    workflow.add_node("translate_response_back", translate_response_back)

    workflow.add_edge(START, "detect_language_and_translate")
    workflow.add_edge("detect_language_and_translate", "query_or_respond")
    workflow.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: "translate_response_back", "tools": "tools"},
    )
    workflow.add_edge("tools", "generate")
    workflow.add_edge("generate", "translate_response_back")
    workflow.add_edge("translate_response_back", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app