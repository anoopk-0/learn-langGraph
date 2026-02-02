"""
Edges in LangGraph define the transitions between nodes, controlling the workflow's path.
You can add direct edges (always go to the next node) or conditional edges (branch based on state).


Types of Edges:
1. Direct Edge: Always goes from one node to another.
2. Conditional Edge: Uses a routing function to decide which node(s) to run next based on the current state.
3. Entry/Exit Edges: Connect the graph's start and end points.

  [START] --> [greet_node] --(if 'search' in message)--> [search_node] --> [END]
                       |
                       +--(else)-----------------------> [END]
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

def greet_node(state: AgentState) -> AgentState:
    new_message = "Hello! How can I assist you today?"
    return {"messages": [new_message]}

def search_node(state: AgentState) -> AgentState:
    user_query = state["messages"][-1] if state["messages"] else ""
    search_result = f"Searching for: '{user_query}'..."
    return {"messages": [search_result]}

def decide_next_node(state: AgentState):
    last_message = state["messages"][-1] if state["messages"] else ""
    if "search" in last_message.lower():
        return "search"
    else:
        return END

workflow = StateGraph(AgentState)
workflow.add_node("greet", greet_node)
workflow.add_node("search", search_node)
workflow.set_entry_point("greet")
workflow.add_conditional_edges(
    "greet",
    decide_next_node,
    {
        "search": "search",
        END: END,
    }
)
workflow.add_edge("search", END)

# Compile and run the graph
app = workflow.compile()

if __name__ == "__main__":
    print("--- Invocation 1 (Greeting) ---")
    result1 = app.invoke({"messages": []})
    print(result1)

    print("\n--- Invocation 2 (Search) ---")
    result2 = app.invoke({"messages": ["I need to search for information about LangChain."]})
    print(result2)
