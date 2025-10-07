## LangGraph Long-Term Memory

Long-term memory in LangGraph enables agents to retain information across multiple conversations, sessions, or users. Unlike short-term memory (thread-scoped checkpoints), long-term memory is organized in custom namespaces and persists beyond individual runs.

### Memory Types in AI Agents
LangGraph supports several memory paradigms inspired by human cognition:

| Memory Type   | What is Stored         | Human Example           | Agent Example                |
|--------------|-----------------------|------------------------|------------------------------|
| Semantic     | Facts                 | School knowledge       | User facts, preferences      |
| Episodic     | Experiences           | Past events            | Agent actions, chat history  |
| Procedural   | Instructions/rules    | Motor skills, instincts| System prompt, agent logic   |

#### Semantic Memory
- Stores facts and concepts (e.g., user profile, preferences).
- Can be managed as a single profile (JSON document) or a collection of documents.
- Profiles are easy to update but may become complex; collections offer higher recall but require careful update logic.
- Semantic memory is not the same as semantic search (retrieval by meaning).

#### Episodic Memory
- Stores experiences and sequences (e.g., chat history, agent actions).
- Often implemented via few-shot example prompting or dynamic retrieval from datasets (e.g., LangSmith).
- Useful for agents that learn from past interactions or need to recall specific events.

#### Procedural Memory
- Stores instructions and rules for agent behavior (e.g., prompts, meta-instructions).
- Can be refined via reflection/meta-prompting, where the agent updates its own instructions based on feedback or recent conversations.



### 📝 Writing and Updating Memories

#### In the Hot Path
- Memories are created/updated during agent execution.
- Immediate availability and transparency for users.
- May increase agent latency and complexity.
- Example: Upserting user facts after each message.

#### In the Background
- Memories are created/updated asynchronously or as background tasks.
- Reduces latency and separates memory logic from main application flow.
- Requires careful scheduling and triggering (e.g., cron jobs, manual triggers).


### Memory Storage in LangGraph
- Long-term memories are stored as JSON documents in a Store (e.g., InMemoryStore, DB-backed store).
- Each memory is organized by namespace (e.g., user ID, application context) and key (unique identifier).
- Namespaces enable hierarchical organization and cross-namespace searching.

```python
from langgraph.store.memory import InMemoryStore

# Example embedding function (replace with a real embedding model in production)
def embed(texts: list[str]) -> list[list[float]]:
    return [[float(i) for i in range(2)] for _ in texts]

# Initialize the in-memory store with vector search capability
dims = 2
store = InMemoryStore(index={"embed": embed, "dims": dims})

# Define a namespace for the user's context
user_id = "user-123"
application_context = "chatbot"
namespace = (user_id, application_context)

# Store a semantic memory (profile/fact)
store.put(
    namespace,
    "profile",
    {
        "name": "Alice",
        "preferences": ["short replies", "English", "Python"],
        "my-key": "my-value",
    },
)

# Store an episodic memory (event)
store.put(
    namespace,
    "event-1",
    {
        "timestamp": "2025-09-10T10:00:00Z",
        "interaction": "Asked about weather",
    },
)

# Retrieve a memory by key
profile = store.get(namespace, "profile")
print("Profile:", profile)

# Search for memories by content and vector similarity
results = store.search(
    namespace, filter={"my-key": "my-value"}, query="preferences"
)
print("Search results:", results)

# List all memories in the namespace
all_memories = store.list(namespace)
print("All memories:", all_memories)

```

### Choosing a Database for LangGraph Long-Term Memory

The best database for LangGraph long-term memory depends on your application's scale, requirements, and infrastructure:

- **InMemoryStore**: Use for prototyping, testing, or small-scale apps. Not persistent—data is lost when the process ends.
- **SQLite**: Use for local apps, small deployments, or when you need simple file-based persistence. Easy to set up, but not ideal for high concurrency or large datasets.
- **Postgres**: Use for production, enterprise, or cloud-native apps. Scalable, reliable, supports advanced features (e.g., vector search with pgvector).
- **MongoDB**: Use if you need flexible schemas, document-oriented storage, or horizontal scaling. Good for distributed systems and unstructured data.
- **Other DBs**: Consider alternatives (e.g., MySQL, Redis, cloud-native stores) if they better fit your tech stack or scaling needs.

**Summary:**
- For prototyping: InMemoryStore or SQLite
- For production: Postgres (recommended), or MongoDB for document flexibility
- Choose based on persistence, scalability, and your data model needs
