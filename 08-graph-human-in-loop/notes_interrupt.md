## LangGraph Interrupt ##

LangGraph supports **human-in-the-loop** workflows by allowing you to pause graph execution, surface data for review, and resume from the same point. There are two main ways to interrupt execution:

### 1. Dynamic Interrupt: `interrupt()`

`interrupt()` lets a node **pause the graph**, surface a payload for human review (or UI), and then **resume** from the exact spot with the human’s response—like `input()` but built for production (non‑blocking, checkpointed, resumable). It relies on LangGraph’s persistence (checkpoints) so the graph can stop indefinitely and continue later with the same `thread_id`. 

- **Dynamic interrupt**: call `interrupt(value)` inside a node to pause based on runtime conditions (e.g., “approve tool call?”).
- **Stateful**: you must compile the graph with a **checkpointer** and run with a **thread_id** to resume.   
- **Resume**: send `Command(resume=...)` to continue; the return value of `interrupt(...)` is the human‑provided data.

```python
from typing import TypedDict
import uuid
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    text: str

def needs_review(state: State):
    revision = interrupt({"to_review": state["text"]})
    return {"text": revision}

g = StateGraph(State)
g.add_node("needs_review", needs_review)
g.add_edge("start", "needs_review")

checkpointer = InMemorySaver()
app = g.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# Initial run: returns interrupt envelope
result = app.invoke({"text": "Draft: hello world"}, config=config)
print(result["__interrupt__"])  # Show this to user/UI

# Resume after human input
final = app.invoke(Command(resume="Edited: Hello, World!"), config=config)
print(final)  # {'text': 'Edited: Hello, World!'}
```

```python
for event in app.stream({"text": "Needs approval"}, config=config):
    if "__interrupt__" in event:
        human_answer = "Approved and revised text"
        break

for event in app.stream(Command(resume=human_answer), config=config):
    pass  # Graph continues from paused node
```

### 2. Static Interrupt: `interrupt_before`

Use `interrupt_before` to pause execution **before** a specific node runs. This is useful for debugging, validation, or manual approval.

**Key Features:**

- Pauses before a node executes.
- Used for inspection, testing, or human validation.
- Resume manually after review.

```python
from langgraph import Graph, interrupt_before

def draft_email(state):
    state["email"] = "Subject: Meeting\nBody: Let's meet at 3pm."
    return state

def approve_email(state):
    # Human approval logic here
    approved = input("Approve email? (y/n): ")
    state["sent"] = approved.lower() == "y"
    return state

graph = Graph()
graph.add_node("Draft", draft_email)
graph.add_node("Approve", approve_email)
graph.set_interrupt(interrupt_before("Approve"))
graph.run()
```

### Comparison Table

| Feature       | `interrupt_before`        | `interrupt()`                 |
| ------------- | ------------------------- | ----------------------------- |
| Type          | Static                    | Dynamic                       |
| Use Case      | Debugging, pre-node pause | Production, async human input |
| Trigger Point | Before node execution     | Anywhere in graph             |
| Resume        | Manual                    | Thread status & persistence   |

**Summary:**

- Use `interrupt()` for dynamic, runtime human-in-the-loop workflows.
- Use `interrupt_before` for static, pre-node pauses (debugging, validation).
- Both rely on LangGraph’s persistence for resumability.
