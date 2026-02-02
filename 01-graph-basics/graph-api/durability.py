"""
Durable execution allows a LangGraph workflow to save its state and resume from where it stopped. This makes workflows reliable even if they are interrupted, paused, or crash.

> Save progress → Stop → Resume later

Durability Modes
- `exit`: Saves state only when the workflow exits. If interrupted, progress since the last exit is lost.
- `async`: Saves state asynchronously during execution. Minimizes lost progress if interrupted, but mayhave slight delays.
- `sync`: Saves state synchronously at each step. Ensures no progress is lost if interrupted, but may slow down execution.

NOTE: langGraph achieves durability through Checkpointers, which handle saving and loading workflow state.

Recommendations:
    - Prefer "sync" durability for critical workflows to avoid data loss.
    - Always configure a checkpointer and unique thread ID.
    - Encapsulate side-effect and non-deterministic operations in nodes for consistent replay.
    - Ensure side-effect operations are idempotent to prevent duplication on retries.
"""

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
