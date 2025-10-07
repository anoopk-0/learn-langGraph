"""
Hybrid memory agent:
- Structured long-term memory: LangGraph InMemoryStore (namespaced key-value, with simple index)
- Vector/semantic memory: Chroma (local vector DB) using Ollama embeddings (nomic-embed-text)
- Short-term checkpoints: InMemorySaver (thread state)

References
----------
- LangGraph Store API & data model: https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/langgraph/store/base/__init__.py
- Store semantic search (concept): https://blog.langchain.com/semantic-search-for-langgraph-memory/
- Chroma getting started: https://docs.trychroma.com/getting-started
- LangChain × Chroma integration: https://python.langchain.com/docs/integrations/vectorstores/chroma/
"""

# Hybrid memory agent using LangGraph InMemoryStore (structured memory) and Chroma (vector/semantic memory)

from typing import TypedDict, Sequence, Annotated, List, Optional
import uuid
from datetime import datetime, timezone
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableConfig

# --- Structured long-term memory (namespaced key-value store) ---
def _toy_embed(texts: list[str]) -> list[list[float]]:
    """Toy embedding function for demonstration. Replace with a real model in production."""
    return [[float(i) for i in range(2)] for _ in texts]

store = InMemoryStore(index={"embed": _toy_embed, "dims": 2})

# --- Vector/semantic memory (Chroma + Ollama embeddings) ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector = Chroma(
    collection_name="hybrid_memories",
    embedding_function=embeddings,
    persist_directory="./chroma_db",  # Persistent local vector DB
)

# --- Agent state definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    memory_context: Annotated[List[str], list]

# --- Initialize LLM ---
llm = ChatOllama(model="llama3.2:3b", temperature=0)

# --- Memory management helpers ---
def _ns(config: RunnableConfig) -> tuple[str, ...]:
    """Get the user/app namespace for memory storage."""
    user_id = config.get("configurable", {}).get("user_id", "anonymous")
    return (user_id, "chatbot")

def ensure_profile(store: BaseStore, ns: tuple[str, ...]) -> None:
    """Ensure a user profile exists in structured memory."""
    if store.get(ns, "profile") is None:
        store.put(ns, "profile", {
            "name": "Alice",
            "preferences": ["short replies", "English", "Python"],
            "my-key": "my-value",
            "type": "profile",
            "ns": "/".join(ns),
        })

def write_event(store: BaseStore, ns: tuple[str, ...], user_msg: str) -> str:
    """Write an episodic event to structured memory."""
    key = f"event-{uuid.uuid4().hex[:8]}"
    value = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interaction": user_msg,
        "type": "event",
        "ns": "/".join(ns),
    }
    store.put(ns, key, value)
    return key

def upsert_chroma(ns: tuple[str, ...], key: str, text: str, metadata: dict) -> None:
    """Upsert an event into Chroma vector DB for semantic search."""
    doc_id = "::".join((*ns, key))
    vector.upsert(ids=[doc_id], documents=[text], metadatas=[metadata])

def recall_from_store(store: BaseStore, ns: tuple[str, ...], query: str, limit: int = 3) -> List[str]:
    """Recall relevant memories from structured store using semantic search and filters."""
    results = store.search(ns, filter={"my-key": "my-value"}, query=query, limit=limit)
    out: List[str] = []
    for r in results:
        v = r.value
        if "preferences" in v:
            prefs = ", ".join(v["preferences"])
            out.append(f"Profile prefs: {prefs} (score={r.score})")
        if v.get("type") == "event":
            out.append(f"Past event: {v['interaction']} at {v['timestamp']} (score={r.score})")
    return out

def recall_from_chroma(ns: tuple[str, ...], query: str, k: int = 3) -> List[str]:
    """Recall relevant memories from Chroma vector DB using semantic similarity."""
    results = vector.similarity_search_with_score(query, k=k, filter={"ns": "/".join(ns)})
    out: List[str] = []
    for doc, score in results:
        kind = (doc.metadata or {}).get("kind")
        if kind == "event":
            out.append(f"Past event: {doc.page_content} (score={score:.3f})")
        else:
            out.append(f"{doc.page_content} (score={score:.3f})")
    return out

# --- Graph nodes ---
def memory_node(state: AgentState, *, store: BaseStore, config: RunnableConfig) -> AgentState:
    """
    - Ensure user profile exists
    - Write current event to structured memory and Chroma
    - Recall hybrid context from both stores
    """
    ns = _ns(config)
    ensure_profile(store, ns)
    last_human: Optional[HumanMessage] = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_human = m
            break
    user_text = last_human.content if last_human else ""
    key = write_event(store, ns, user_text)
    upsert_chroma(ns, key, f"event: {user_text}", {"kind": "event", "ns": "/".join(ns)})
    store_snips = recall_from_store(store, ns, user_text, limit=3)
    chroma_snips = recall_from_chroma(ns, user_text, k=3)
    return {"memory_context": store_snips + chroma_snips}

def llm_chat_node(state: AgentState) -> AgentState:
    """
    - Build system prompt with recalled memory context
    - Invoke LLM and append response
    """
    mem_text = "\n".join(state.get("memory_context", [])) or "No prior memories."
    system_prompt = SystemMessage(
        content=(
            "You are a helpful assistant. Use the following user-specific memory when relevant.\n"
            f"--- MEMORY ---\n{mem_text}\n"
            "Summarize the user's request before answering."
        )
    )
    response: AIMessage = llm.invoke([system_prompt] + list(state["messages"]))
    return {"messages": [response]}

# --- Build and run the workflow ---
workflow = StateGraph(AgentState)
workflow.add_node("memory", memory_node)
workflow.add_node("chat", llm_chat_node)
workflow.add_edge(START, "memory")
workflow.add_edge("memory", "chat")
workflow.add_edge("chat", END)

checkpointer = InMemorySaver()
agent = workflow.compile(checkpointer=checkpointer, store=store)

if __name__ == "__main__":
    print("Type 'exit' to quit.")
    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() == "exit":
            break
        config = {"configurable": {"thread_id": "1", "user_id": "user-123"}}
        result = agent.invoke({"messages": [HumanMessage(content=user_input)], "memory_context": []}, config=config)
        if result and "messages" in result and result["messages"]:
            print("Assistant:", result["messages"][-1].content)
