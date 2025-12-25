## Subgraphs

A subgraph is a graph that can be embedded as a node within another graph, bringing encapsulation to LangGraph workflows. Subgraphs help organize complex systems by allowing you to compose multiple graphs as modular components.

**Benefits of Using Subgraphs:**

- Facilitate the construction of multi-agent systems.
- Enable reuse of node collections across different graphs.
- Allow teams to independently develop and maintain separate graph components, provided the subgraph interface is followed.

![subgraph](../../images/subgraph.png)

### Communication Between Parent Graph and Subgraph

The main question when adding subgraphs is how the parent graph and subgraph communicate, i.e. how they pass the state between each other during graph execution. There are two scenarios:

#### 1. Shared State Keys

Parent and subgraph have shared state keys in their state schemas. In this case, you can include the subgraph as a node in the parent graph.

```python
from langgraph.graph import StateGraph, MessagesState, START

# Subgraph
def call_model(state: MessagesState):
	response = model.invoke(state["messages"])
	return {"messages": response}

subgraph_builder = StateGraph(MessagesState)
subgraph_builder.add_node(call_model)
subgraph = subgraph_builder.compile()

# Parent graph
builder = StateGraph(MessagesState)
builder.add_node("subgraph_node", subgraph)
builder.add_edge(START, "subgraph_node")
graph = builder.compile()

# Usage
graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
```

#### 2. Different State Schemas

Parent graph and subgraph have different schemas (no shared state keys). You have to call the subgraph from inside a node in the parent graph. This is useful when you need to transform state before or after calling the subgraph.

```python
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.graph.message import add_messages

class SubgraphMessagesState(TypedDict):
	subgraph_messages: Annotated[list[AnyMessage], add_messages]

# Subgraph
def call_model(state: SubgraphMessagesState):
	response = model.invoke(state["subgraph_messages"])
	return {"subgraph_messages": response}

subgraph_builder = StateGraph(SubgraphMessagesState)
subgraph_builder.add_node("call_model_from_subgraph", call_model)
subgraph_builder.add_edge(START, "call_model_from_subgraph")
subgraph = subgraph_builder.compile()

# Parent graph
def call_subgraph(state: MessagesState):
	response = subgraph.invoke({"subgraph_messages": state["messages"]})
	return {"messages": response["subgraph_messages"]}

builder = StateGraph(MessagesState)
builder.add_node("subgraph_node", call_subgraph)
builder.add_edge(START, "subgraph_node")
graph = builder.compile()

# Usage
graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
```
