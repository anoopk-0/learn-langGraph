"""
Durable execution in LangGraph enables workflows to persist their state at key points, allowing them to pause and resume reliably. This is essential for long-running tasks, human-in-the-loop scenarios, and robust error recovery.

Durability Modes:
    - "exit": Saves state only at workflow completion (success or error). Fastest, but progress is lost if interrupted mid-execution. Suitable for short, non-critical workflows. It is the default mode.
    - "async": Saves state asynchronously during execution. Balances speed and reliability, but may lose recent progress if a crash occurs before the save completes.
    - "sync": Saves state synchronously before each step. Most reliable—every step is checkpointed. Recommended for production and critical workflows.

Why Use Durability?
    - Enables recovery from failures, interruptions, or manual pauses.
    - Supports human-in-the-loop workflows: users can inspect or modify state before resuming.
    - Prevents loss of progress in long-running or expensive operations.

Production Best Practices:
    - Prefer "sync" durability for critical workflows to avoid data loss.
    - Always configure a checkpointer and unique thread ID.
    - Encapsulate side-effect and non-deterministic operations in nodes for consistent replay.
    - Ensure side-effect operations are idempotent to prevent duplication on retries.

Summary:
    - Durability modes let you choose between performance and reliability. For production, "sync" is recommended. LangGraph’s persistence ensures workflows are robust, resumable, and fault-tolerant.
"""
# Example (StateGraph with Durability Modes)
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
import uuid

class State(TypedDict):
    url: str
    result: str

def call_api(state: State):
    # Simulate an API call
    import requests
    result = requests.get(state['url']).text[:100]
    return {"result": result}

# Build the graph
builder = StateGraph(State)
builder.add_node("call_api", call_api)
builder.add_edge(START, "call_api")
builder.add_edge("call_api", END)

# Enable persistence with a checkpointer
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Use a thread ID to track workflow instance
thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}

# Run with durability mode
result = graph.invoke(
    {"url": "https://www.example.com"},
    config,
    durability="sync"  # Options: "exit", "async", "sync"
)

# Or stream events with durability
for event in graph.stream(
    {"url": "https://www.example.com"},
    config,
    durability="async"
):
    print(event)
