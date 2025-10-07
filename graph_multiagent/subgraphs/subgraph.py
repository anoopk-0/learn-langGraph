from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
import os

class ChildState(TypedDict):
    messages: Annotated[list, add_messages]

os.environ["TAVILY_API_KEY"] = "your_tavily_api_key_here"

llm = ChatOllama(model="llama3.2:3b", temperature=0)
tavily_search = TavilySearch(max_results=2)
tools = [tavily_search]
llm_with_tools = llm.bind_tools(tools=tools)

def agent(state: ChildState):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])],
    }

def tools_router(state: ChildState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        return END

tool_node = ToolNode(tools=tools)

subgraph = StateGraph(ChildState)
subgraph.add_node("agent", agent)
subgraph.add_node("tool_node", tool_node)
subgraph.set_entry_point("agent")
subgraph.add_conditional_edges("agent", tools_router)
subgraph.add_edge("tool_node", "agent")
search_app = subgraph.compile()


print("-------------------------------Shared Schema (Direct Embedding)----------------------------------")
# Case 1: Shared Schema (Direct Embedding)
from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, START, END
from langchain_core.messages import HumanMessage


class ParentState(TypedDict):
    messages: Annotated[list, add_messages]

parent_graph = StateGraph(ParentState)

"""
NOTE:

- The subgraph `search_app` uses the same state schema (`messages`) as the parent graph.
- This allows direct embedding of the subgraph without any state transformation.
"""
parent_graph.add_node("search_agent", search_app)

# Connect the flow
parent_graph.add_edge(START, "search_agent")
parent_graph.add_edge("search_agent", END)

# Compile parent graph
parent_app = parent_graph.compile()

# Run the parent graph
result = parent_app.invoke({"messages": [HumanMessage(content="How is the weather in Chennai?")]})
print(result)


#Different Schema (Invoke with Transformation)
print("-------------------------------Different Schema (With Transformation)----------------------------------")
from typing import TypedDict, Annotated, Dict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

# Define parent graph with different schema
class QueryState(TypedDict):
    query: str
    response: str

# Node function: transforms parent state to subgraph state, invokes subgraph, and transforms result back
def search_agent(state: QueryState) -> Dict:
    # Transform parent schema to subgraph schema
    subgraph_input = {
        "messages": [HumanMessage(content=state["query"])]
    }
    # Invoke the subgraph with transformed input
    subgraph_result = search_app.invoke(subgraph_input)
    # Transform subgraph result back to parent schema
    assistant_message = subgraph_result["messages"][-1]
    return {"response": assistant_message.content}

parent_graph = StateGraph(QueryState)
parent_graph.add_node("search_agent", search_agent)
parent_graph.add_edge(START, "search_agent")
parent_graph.add_edge("search_agent", END)
parent_app = parent_graph.compile()

result = parent_app.invoke({"query": "How is the weather in Chennai?", "response": ""})
print(result)
