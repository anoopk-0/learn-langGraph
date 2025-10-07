## Context Engineering in LangGraph

Context engineering involves designing systems that deliver relevant information and resources to AI applications in the optimal format for task completion. In LangGraph, context is defined along two main dimensions:

- `Mutability`: Whether the data is static (unchanging) or dynamic (evolving during execution).
- `Lifetime`: Whether the data is scoped to a single run or persists across multiple sessions.

### Runtime Context vs LLM Context

- **Runtime context**: Refers to all data, resources, and dependencies available to your application during execution. This includes static configuration (e.g., user metadata, API keys, tool references), dynamic state (e.g., conversation history, intermediate results), and persistent stores (e.g., user profiles, historical interactions). Runtime context is managed by LangGraph and determines what information is accessible to your code and agents at any point in a run. It is used to decide what should be included in prompts, what tools are available, and how the application behaves.

- **LLM context**: Represents the specific information that is packaged into the prompt and sent to the language model (LLM) for processing. This typically includes the current conversation history, relevant facts, instructions, and any additional context needed for the LLM to generate an appropriate response. LLM context is a subset of the runtime context, carefully selected and formatted to fit within the model's input constraints.

- **Context window**: The maximum number of tokens (words, punctuation, and special symbols) that the LLM can process in a single input. The context window limits how much information can be included in the prompt. Effective context engineering involves selecting and compressing the most relevant data from the runtime context to fit within the context window, ensuring the LLM receives all necessary information without exceeding token limits.

### LangGraph Context Management

LangGraph provides flexible mechanisms for managing different types of context, each with distinct storage and persistence requirements. Choosing the appropriate storage solution depends on whether your context is static, dynamic within a single run, or dynamic across multiple sessions.

#### 1. Static Runtime Context

- **Description:** Immutable data provided at the start of a run, such as configuration settings, environment variables, secrets, or tool references.
- **Storage:** No database is required. Static context is typically loaded from config files, environment variables, or secret managers and passed directly to the application.
- **Use Cases:** User metadata, API keys, tool configurations.

#### 2. Dynamic Runtime Context (State)

- **Description:** Mutable data that evolves during a single execution, including conversation history, intermediate results, and temporary variables.
- **Storage:** Usually managed in-memory for fast access and low latency. Common approaches include Python dictionaries, custom state objects, or ephemeral caches.
- **Persistence:** By default, state is not persisted beyond the current run. If you need to persist state for debugging, recovery, or analysis, you can use temporary databases like Redis or local file storage.
- **Use Cases:** Tracking conversation turns, storing intermediate computation results, managing temporary workflow state.

#### 3. Dynamic Cross-Conversation Context (Store)

- **Description:** Persistent, mutable data that spans multiple runs or sessions, such as user profiles, preferences, and historical interactions.
- **Storage:** Requires a persistent database to ensure data durability and accessibility across sessions.
- **Recommended Databases:**
  - **MongoDB:** A flexible document-oriented database, ideal for storing user profiles, preferences, and historical data.
  - **ChromaDB:** A vector database designed for storing embeddings and supporting retrieval-augmented generation (RAG) workflows.
  - **SQL Databases (e.g., PostgreSQL, MySQL):** Suitable for structured, relational data with complex queries and relationships.
  - **Other NoSQL Databases:** Useful for scalable, schema-less storage needs.
- **Use Cases:** Maintaining long-term user memory, enabling personalization, supporting multi-agent workflows, and facilitating data sharing between sessions.

By understanding the nature of your context and its lifecycle, you can select the most appropriate storage solution to optimize performance, scalability, and reliability in your LangGraph applications.

### Static Runtime Context Example

Static runtime context consists of immutable data provided at the start of a run, such as configuration settings or environment variables. No database is required; values are typically loaded from config files or environment variables.

```python
from dataclasses import dataclass

@dataclass
class ContextSchema:
    user_name: str

# Provide static context when invoking the graph
graph.invoke(
    {"messages": [{"role": "user", "content": "hi!"}]},
    context={"user_name": "John Smith"}
)
```

To use static context in agent prompts, access it via the runtime object:

```python
def prompt(state):
    runtime = get_runtime(ContextSchema)
    system_msg = f"You are a helpful assistant. Address the user as {runtime.context.user_name}."
    return [{"role": "system", "content": system_msg}] + state["messages"]
```

### Dynamic Runtime Context (State)

Dynamic runtime context refers to mutable data that changes during a single execution, managed via the LangGraph state object. This includes conversation history, intermediate results, and values produced by tools or LLM outputs. The state object serves as short-term memory for the duration of a run.

```python
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph

class CustomState(TypedDict):
    messages: list[AnyMessage]
    extra_field: int

def node(state: CustomState):
    # Access and update state fields
    updated_extra = state["extra_field"] + 1
    return {
        "extra_field": updated_extra
    }

builder = StateGraph(CustomState)
builder.add_node(node)
builder.set_entry_point("node")
graph = builder.compile()
```

**NOTE:**

- State is limited to a single run and is not persisted unless explicit memory is enabled.
- For persisting state across runs, refer to the LangGraph memory documentation.

### Dynamic Cross-Conversation Context (Store)

Dynamic cross-conversation context refers to persistent, mutable data that is maintained across multiple sessions or conversations. In LangGraph, this is managed using a store, which serves as long-term memory for your application. Typical examples include user profiles, preferences, and historical interactions.

**Common Use Cases:**

- Accessing and updating persistent user data (e.g., profiles, preferences, interaction history)
- Enabling data sharing between agents or workflows
- Supporting personalization and continuity across sessions

The store ensures that relevant information is available beyond a single run, allowing your application to deliver consistent and personalized experiences.
