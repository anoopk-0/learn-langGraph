"""
In LangGraph, a node is a function that performs a unit of work in the workflow. Each node receives the current state, does some logic or side-effect, and returns an updated state.

Nodes can represent agent actions, tool calls, or any computation.

NOTE: A Node in LangGraph is a function that transforms state and represents one step in your AI workflow.
"""

from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# node 1: Greets the user
def greet_node(state: AgentState) -> AgentState:
    """
    This node adds a greeting message to the state.
    Nodes always take the state as input and return an updated state.
    """
    new_message = "Hello! How can I assist you today?"
    return {"messages": [new_message]}

# node 2: Processes a search query
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

"""
In LangGraph, a ToolNode is a special type of node designed to execute tools (functions, APIs, or utilities) that an LLM decides to call.
"""

from langgraph.prebuilt import ToolNode
from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
    return a + b

tool_node = ToolNode([add])
