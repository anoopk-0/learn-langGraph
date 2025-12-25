## LangGraph Persistence

LangGraph’s persistence system is designed to enable robust, stateful, and resilient multi-actor applications. It provides both short-term and long-term memory, supports advanced workflows, and integrates with various storage backends. Here’s a detailed breakdown:

- Short-term memory (Checkpointing):  
    At every step of a workflow, LangGraph can capture the entire state of the graph, including configuration, metadata, values, and execution tasks. This enables features like pausing and resuming execution, replaying previous steps, and recovering from failures. Each run is tracked by a unique thread ID, and checkpoints are stored as snapshots that can be inspected or updated. This architecture supports human-in-the-loop workflows, debugging, and fault tolerance.

- Long-term memory (Store):  
    Beyond individual runs, LangGraph can persist information across sessions and threads. This is useful for storing user profiles, conversation histories, embeddings, and other data that needs to be retained and retrieved over time. Data is organized into namespaces, allowing for separation and sharing across users or contexts. The store system supports semantic search and vector similarity, enabling retrieval-augmented generation and personalization.

- Flexible storage backends:  
    LangGraph supports multiple storage backends for both checkpointing and store systems. Options include in-memory (fast, ephemeral), SQLite (file-based, easy setup), PostgreSQL (scalable, production-ready), and custom integrations like MongoDB for document-oriented storage. This flexibility allows developers to choose the backend that best fits their scale, reliability, and data model requirements.

- Serialization and compatibility:  
    To handle complex data types, LangGraph uses a robust serialization layer that supports JSON, msgpack, and pickle. This ensures compatibility with a wide range of Python objects, including NumPy arrays and Pydantic models.

### 🔧 Core Persistence Architecture

LangGraph’s persistence system is organized into two primary subsystems:

#### 1. Checkpointing System (Short-term Memory)
- **Purpose:** Records the graph’s state at each step, enabling recovery, replay, and inspection.
- **Mechanism:**
    - Each workflow run is identified by a unique `thread_id`.
    - At every step, a checkpoint (as a `StateSnapshot` object) is saved, containing:
        - `config`: Execution settings
        - `metadata`: Step details, errors, timing
        - `values`: State channel data
        - `next`: Upcoming nodes
        - `tasks`: Execution tasks and errors
- **Access Methods:**
    - Get latest state: `graph.get_state(config)`
    - View full history: `graph.get_state_history(config)`
    - Replay/update: `graph.update_state(config, values)`
- **Advantages:**
    - Human-in-the-loop: Pause, inspect, and resume workflows
    - Fault tolerance: Recover from last successful step
    - Time travel: Replay or branch from any checkpoint

#### 2. Store System (Long-term Memory)
- **Purpose:** Persists data across sessions and threads, such as user profiles, conversation logs, and embeddings.
- **Mechanism:**
    - Data is organized by namespaces (e.g., `(user_id, "memories")`)
    - Supports semantic search and vector similarity
    - Accessible and updatable from any node
- **Advantages:**
    - Share data across threads and users
    - Enable retrieval-augmented generation and personalization

### 🗃️ Storage Backends: Choosing the Right Option

LangGraph offers several storage backends for both checkpointing and store systems. Select based on your requirements:

#### Checkpointing Backends
- `InMemorySaver:` Fast, non-persistent; best for development and testing.
- `SqliteSaver / AsyncSqliteSaver:` File-based, easy setup; suitable for local or small-scale use.
- `PostgresSaver / AsyncPostgresSaver:` Scalable, concurrent, production-ready; ideal for cloud and enterprise.
- `MongoDB (Custom):` Not built-in, but possible via custom checkpointer for document storage and scaling.


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

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": ""}, config)
```


#### Store Backends
- `InMemoryStore:` Fast, simple; for prototyping, not persistent.
- `SqliteStore:` File-based, supports vector search; good for local apps.
- `PostgresStore:` Advanced, supports pgvector; recommended for production.
- `MongoDBStore (Custom):` Flexible schema, distributed storage; suitable for large-scale, document-oriented needs.


```python
from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()
user_id = "1"
namespace = (user_id, "memories")
memory_id = str(uuid.uuid4())
memory = {"food_preference": "I like pizza"}
in_memory_store.put(namespace, memory_id, memory)
memories = in_memory_store.search(namespace)
print(memories[-1].dict())
```

Backend Selection Guide
- `Development/Testing:` Use InMemorySaver/InMemoryStore for speed and simplicity.
- `Local/Small-scale:` Use SqliteSaver/SqliteStore for easy setup and persistence.
- `Production/Enterprise:` Use PostgresSaver/PostgresStore for scalability and advanced features.
- `Flexible/NoSQL:` Consider MongoDB for document storage and horizontal scaling (requires custom integration).

### 🔄 Serialization Layer

**Serialization** refers to transforming Python objects—such as lists, dictionaries, NumPy arrays, or Pydantic models—into formats suitable for storage or transmission (like JSON, msgpack, or pickle). LangGraph’s serialization system enables these objects to be saved and restored across various storage backends, ensuring workflow compatibility and resilience.
- LangGraph uses **JsonPlusSerializer** to support complex Python objects (NumPy arrays, dataclasses, Pydantic models).
- Provides compatibility via `msgpack`, `JSON`, and `pickle` as needed.
