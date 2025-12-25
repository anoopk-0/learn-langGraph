"""
LangGraph Looping Pattern

The looping pattern allows a graph to repeatedly execute nodes until a condition is met.
This is useful for tasks like retries, iterative processing, or agent workflows that need to revisit steps.

How it works:
- After a node runs, a routing function checks the state to decide whether to loop back or proceed to the next node.
- The graph can loop any number of times until the exit condition is satisfied.
"""

from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
import random

class AgentState(TypedDict):
    name: str
    numbers: List[int]
    counter: int

# Node: Greets the user and initializes the counter
def greeting_node(state: AgentState) -> AgentState:
    """
    Adds a greeting to the name and initializes the counter.
    """
    state['name'] = f"Hey {state['name']}, how is your day going?"
    state['counter'] = 0
    return state

# Node: Adds a random number to the list and increments the counter
def random_number_node(state: AgentState) -> AgentState:
    """
    Appends a random integer to the numbers list and increments the counter.
    """
    state['numbers'].append(random.randint(1, 10))
    state['counter'] += 1
    return state

# Node: Decides whether to continue looping or finish
def should_continue_node(state: AgentState) -> str:
    """
    Returns the edge name to continue looping or finish based on the counter.
    """
    if state['counter'] < 5:
        return "continue_edge"
    else:
        return "finish_edge"

# Build the workflow graph
graph = StateGraph(AgentState)
graph.add_node("greeting_node", greeting_node)
graph.add_node("random_number_node", random_number_node)
graph.add_node("should_continue_router", lambda state: state)  # Routing node for loop control


graph.add_edge("greeting_node", "random_number_node")
graph.add_edge("random_number_node", "should_continue_router")
graph.add_conditional_edges("should_continue_router", should_continue_node, {
    "continue_edge": "random_number_node",  # Loop again
    "finish_edge": END  # Finish
})
graph.set_entry_point("greeting_node")

app = graph.compile()

# Invoke the application with an initial state
response = app.invoke(AgentState(name="Alice", numbers=[], counter=6))
print(response['name'])  # Output: "Hey Alice, how is your day going?"
print("Random numbers generated:", response['numbers'])

# Print graph as Mermaid text (copy to https://mermaid.live/ for visualization)
print(app.get_graph().draw_mermaid())