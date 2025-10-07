"""
# In LangGraph, a node is a function that performs a unit of work in the workflow.
# Each node receives the current state, does some logic or side-effect, and returns an updated state.
# Nodes can represent agent actions, tool calls, or any computation.
# You add nodes to the graph and connect them with edges to define the workflow steps.

"""
from typing import TypedDict, Annotated
import operator

# Example state schema for nodes to use
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# Example node: Greets the user
def greet_node(state: AgentState) -> AgentState:
    """
    This node adds a greeting message to the state.
    Nodes always take the state as input and return an updated state.
    """
    new_message = "Hello! How can I assist you today?"
    return {"messages": [new_message]}

# Example node: Processes a search query
def search_node(state: AgentState) -> AgentState:
    """
    This node simulates a search based on the last user message.
    Nodes can contain any logic, including tool calls or LLM queries.
    """
    user_query = state["messages"][-1] if state["messages"] else ""
    search_result = f"Searching for: '{user_query}'..."
    return {"messages": [search_result]}

# LangGraph Flow Sample (Graph Form)
#
#   [greet_node] ---> [search_node] ---> END
#         |                ^
#         |                |
#         +----------------+
#
# Flow:
# 1. Start at greet_node (greets the user).
# 2. If the user's message contains 'search', go to search_node.
# 3. search_node processes the query and ends the flow.
# 4. If no search is needed, finish after greet_node.
#
# Nodes are functions; edges define transitions.