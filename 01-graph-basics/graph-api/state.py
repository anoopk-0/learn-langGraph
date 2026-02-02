"""
State is a shared data structure that holds the current information or context of the entire application or workflow. In simple terms, it is like a big dictionary where you can store and retrieve data that different parts of your application need to access or modify.
"""

# State is usually defined using TypedDict (for key/value pairs) or dataclass/Pydantic for more features.

from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages

# Example 1: Basic State Definition

class BasicState(TypedDict):
    count: int  # Example: simple counter

# Example 2: More Complex State

class ComplexState(TypedDict):
    count: int
    messages: Annotated[list[HumanMessage | AIMessage], add_messages]  # Stores conversation history with reducer


"""
NOTE ON "add_messages":

add_messages is usually a reducer function. A reducer controls how new values are combined with old values. Instead of replacing messages.

It appends new messages to the existing list, preserving conversation history.
"""
