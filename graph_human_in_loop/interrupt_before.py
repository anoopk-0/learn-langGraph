from langgraph.graph import StateGraph, START, END, add_messages
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

memory = MemorySaver()

document_content = ""  

@tool
def save(filename: str) -> str:
    """
    Saves the current document content to a text file with the given filename.
    If the filename does not end with '.txt', it appends the extension.
    Returns a confirmation message upon successful save.
    """
    global document_content
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"
    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    except Exception as e:
        return f"Error saving document: {str(e)}"


tools = [save]

# Initialize the LLM and bind tools for tool calling
llm = ChatOllama(model="llama3.2:3b", temperature=0).bind_tools(tools=tools)

# Define the state structure for the graph
class BasicState(TypedDict):
    messages: Annotated[List, add_messages]

# Node: LLM model invocation
def model(state: BasicState):
    """
    Calls the LLM with the current messages and appends the response.
    """
    return {
        "messages": [llm.invoke(state["messages"])]
    }

# Node router: Determines next step based on tool calls
def tools_router(state: BasicState):
    """
    Checks if the last message contains tool calls.
    If so, routes to the 'tools' node; otherwise, ends the workflow.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
    else:
        return END

# Build the workflow graph
graph = StateGraph(BasicState)
graph.add_node(model, "model")  # Node for LLM generation
graph.add_node("tool_node", ToolNode(tools=tools))  # Node for tool execution

graph.set_entry_point("model")  # Set entry point for the workflow

graph.add_conditional_edges("model", tools_router)  # Route based on tool calls
graph.add_edge("tool_node", "model")  # Loop back to model after tool execution

# Compile the app with interrupt_before on 'tool_node' for human-in-the-loop
# The 'interrupt_before' argument allows you to pause execution before specified nodes.
# This is useful for human-in-the-loop workflows, enabling manual intervention or review.
# You can specify any node name (not just tool nodes) in the interrupt_before list.
# Here, we interrupt before the 'tool_node', but you could also interrupt before other nodes (e.g., 'model').
app = graph.compile(checkpointer=memory, interrupt_before=["tool_node"])

# Configuration for thread persistence
config = {"configurable": {
    "thread_id": 1
}}

# Start the workflow with an initial human message
events = app.stream({
    "messages": [HumanMessage(content="write notes on LLM?")]
}, config=config, stream_mode="values")

# Print each event's last message in a readable format
for event in events:
    event["messages"][-1].pretty_print()