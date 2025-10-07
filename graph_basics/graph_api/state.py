# LangGraph State Notes
#
# State is a shared data structure that holds the current context, variables, and workflow data.
# All nodes and edges in a LangGraph workflow read from and write to the state.
#
# State is usually defined using TypedDict (for key/value pairs) or dataclass/Pydantic for more features.


from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages

# Example 1: Basic State Definition
#
# TypedDict lets you define a schema for your state as key/value pairs.
# Each key is a variable in the workflow state.
class BasicState(TypedDict):
    count: int  # Example: simple counter

# Example 2: More Complex State
#
# You can add more fields to track additional data, like messages.
# Annotated lets you attach a reducer function to a field.
# Reducers control how updates are applied (e.g., add_messages appends or updates messages).
class ComplexState(TypedDict):
    count: int
    messages: Annotated[list[HumanMessage | AIMessage], add_messages]  # Stores conversation history with reducer
    
    
"""
NOTE ON "add_messages":
add_messages: is a reducer function used in LangGraph to manage lists of messages in your graph state. When you annotate a state field (like messages) with add_messages, it controls how updates to that field are applied.

How it works:

- When a node returns new messages, add_messages appends them to the existing list in the state.
- If a message with the same ID already exists, add_messages updates (overwrites) that message instead of adding a duplicate.
- It also deserializes messages into LangChain Message objects, so you can work with them easily.
- This is useful for tracking conversation history, ensuring new messages are added and existing ones are updated correctly, especially in chat or agent workflows.
"""