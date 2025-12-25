# Example: Using LangGraph's InMemoryStore for long-term memory with semantic search

from typing import TypedDict, Sequence, Annotated, List, Optional
import uuid
from datetime import datetime, timezone
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langchain_ollama import ChatOllama, OllamaEmbeddings

# You can use OllamaEmbeddings for semantic search in production:
# embeddings = OllamaEmbeddings(model="llama3.2:3b")
# store = InMemoryStore(index={"embed": embeddings.embed_documents, "dims": embeddings.dims})

# For demonstration, use a toy embedding function
def embed(texts: list[str]) -> list[list[float]]:
    return [[float(i) for i in range(2)] for _ in texts]

dims = 2
store = InMemoryStore(index={"embed": embed, "dims": dims})

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    memory_context: Annotated[List[str], list]

llm = ChatOllama(model="llama3.2:3b", temperature=0)

# Helpers for memory management

def _user_ns(config: RunnableConfig) -> tuple[str, ...]:
    user_id = config.get("configurable", {}).get("user_id", "anonymous")
    return (user_id, "chatbot")

def ensure_profile(store: BaseStore, ns: tuple[str, ...]) -> None:
    if store.get(ns, "profile") is None:
        store.put(ns, "profile", {
            "name": "Alice",
            "preferences": ["short replies", "English", "Python"],
            "my-key": "my-value",
            "type": "profile"
        })

def write_episodic_event(store: BaseStore, ns: tuple[str, ...], user_msg: str) -> None:
    store.put(ns, f"event-{uuid.uuid4().hex[:8]}", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interaction": user_msg,
        "type": "event"
    })

def recall_memories(store: BaseStore, ns: tuple[str, ...], query: str, limit: int = 3) -> list[str]:
    results = store.search(ns, filter={"my-key": "my-value"}, query=query, limit=limit)
    snippets = []
    for r in results:
        val = r.value
        if "preferences" in val:
            pref = ", ".join(val["preferences"])
            snippets.append(f"Profile prefs: {pref} (score={r.score})")
        if "interaction" in val:
            snippets.append(f"Past event: {val['interaction']} at {val['timestamp']} (score={r.score})")
    return snippets

# Graph nodes

def memory_node(state: AgentState, *, store: BaseStore, config: RunnableConfig) -> AgentState:
    ns = _user_ns(config)
    ensure_profile(store, ns)
    last_human: Optional[HumanMessage] = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_human = m
            break
    user_text = last_human.content if last_human else ""
    write_episodic_event(store, ns, user_text)
    memory_context = recall_memories(store, ns, user_text, limit=3)
    return {"memory_context": memory_context}

def llm_chat_node(state: AgentState) -> AgentState:
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

# Build the graph
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
        user_input = input("User: ").strip()
        if user_input.lower() == "exit":
            break
        config = {
            "configurable": {
                "thread_id": "1",
                "user_id": "user-123"
            }
        }
        result = agent.invoke({"messages": [HumanMessage(content=user_input)], "memory_context": []}, config=config)
        if result and "messages" in result and result["messages"]:
            print("Assistant:", result["messages"][-1].content)

# Recommended pattern
# LangGraph Store for structured long-term memory (PostgresStore in prod).
# Vector DB (Chroma, Weaviate, etc.) for large-scale semantic recall if needed.
# Checkpointer (PostgresSaver or SqliteSaver) for short-term thread state.