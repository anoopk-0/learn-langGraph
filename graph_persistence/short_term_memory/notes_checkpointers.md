## LangGraph Checkpointers

LangGraph’s persistence layer, implemented via checkpointers, acts as a form of short-term memory. It saves the state of a graph at each step within a single thread or workflow. This allows you to:
- Access the current and previous states during execution
- Replay or resume from any checkpoint
- Enable features like fault-tolerance and human-in-the-loop

This memory is temporary and specific to the current thread. When a new thread is started, it begins with a fresh state unless restored from a previous checkpoint. Short-term memory is ideal for managing context and state within a single conversation or workflow.


**Checkpointer**:
- A checkpointer is an object that saves a **checkpoint** (snapshot) of the graph’s state after each step.
- Each checkpoint is tied to a **thread ID**, representing a unique execution context.
- Checkpoints are stored as `StateSnapshot` objects, which include:
  - `config`: Configuration used for the run
  - `metadata`: Execution details (step, errors, timing)
  - `values`: State channel values
  - `next`: Next nodes to execute
  - `tasks`: Execution tasks and errors

**why Use Checkpointers?**
-  `Fault-tolerance`:  If a node fails, the graph can resume from the last successful checkpoint without restarting the entire workflow

-  `Human-in-the-loop`:    Developers can pause execution, inspect the state, modify values, and resume—ideal for debugging and interactive workflows.

-  `Time travel`:  Replay or fork execution from any historical checkpoint, enabling experimentation and branching logic.

-  `Debugging`:    Inspect historical states and transitions to trace bugs or performance bottlenecks

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}

workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

# Create a checkpointer (in-memory for this example)
checkpointer = InMemorySaver()

# Compile the graph with the checkpointer
graph = workflow.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": ""}, config)
```

### Checkpointer Backend Options in LangGraph

LangGraph offers several backend implementations for checkpointers, allowing you to choose the best fit for your environment and requirements. Each backend adheres to the `BaseCheckpointSaver` interface, providing essential methods such as `put()`, `get_tuple()`, `list()`, and `delete_thread()` for managing checkpoints.

| Backend                        | Description                        | Best Use Case                      |
|---------------------------------|------------------------------------|------------------------------------|
| **InMemorySaver**               | Fast, stores checkpoints in RAM; ephemeral (data lost on restart) | Rapid prototyping, development, and testing |
| **SqliteSaver / AsyncSqliteSaver** | File-based, lightweight, easy to set up | Local applications, small-scale deployments |
| **PostgresSaver / AsyncPostgresSaver** | Scalable, robust, supports concurrent access and large datasets | Enterprise, production, cloud-native apps |
| **Custom (e.g., MongoDB)**      | Extendable via abstract base class; supports flexible schemas and document-oriented storage | Custom needs, distributed systems, document storage |

**Tip:**
- Use `InMemorySaver` for speed and simplicity during development.
- Switch to `SqliteSaver` for persistent local storage.
- Choose `PostgresSaver` for reliability and scalability in production.
- Implement a custom backend (e.g., MongoDB) if you need advanced features like flexible schemas or distributed storage.

All backends provide a consistent API for checkpoint management, making it easy to swap implementations as your project grows.

