"""
Defines the structure and workflow of a LangGraph graph, including nodes, state, edges, and conditional logic.

A Graph in LangGraph organizes the flow of data and decisions, showing the sequence and conditional paths between operations.
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# Node: Greets the user
def greet_node(state: AgentState) -> AgentState:
    """
    Returns a greeting message as the first step in the workflow.
    """
    new_message = "Hello! How can I assist you today?"
    return {"messages": [new_message]}

# Node: Processes a search query
def search_node(state: AgentState) -> AgentState:
    """
    Processes the last user message as a search query and returns a search result.
    """
    user_query = state["messages"][-1] if state["messages"] else ""
    # In a real application, this would involve calling a search tool
    search_result = f"Searching for: '{user_query}'..."
    return {"messages": [search_result]}

def decide_next_node(state: AgentState):
    """
    Determines the next node based on the last message content.
    If the message contains 'search', route to the search node; otherwise, end.
    """
    last_message = state["messages"][-1] if state["messages"] else ""
    if "search" in last_message.lower():
        return "search"
    else:
        return END

# StateGraph: Main class for building a graph workflow.
# Pass the state schema (AgentState) to define the structure of shared state.
workflow = StateGraph(AgentState)

workflow.add_node("greet", greet_node)   # Add a node for greeting
workflow.add_node("search", search_node) # Add a node for search
workflow.set_entry_point("greet")        # Set the starting node
workflow.add_conditional_edges(
    "greet",
    decide_next_node,
    {
        "search": "search",  # If decide_next_node returns 'search', go to search node
        END: END,               # If decide_next_node returns END, finish
    }
)
workflow.add_edge("search", END)         # After search, finish

# compile: Checks and finalizes the graph structure before use. Must be called before running the graph.
app = workflow.compile()

# invoke: Runs the graph with a given input state. Returns the final state after execution.
print("--- Invocation 1 (Greeting) ---")
result1 = app.invoke({"messages": []})
print(result1)

print("\n--- Invocation 2 (Search) ---")
result2 = app.invoke({"messages": ["I need to search for information about LangChain."]})
print(result2)