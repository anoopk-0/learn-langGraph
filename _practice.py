from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    messages: Annotated[list[HumanMessage | AIMessage], add_messages]
