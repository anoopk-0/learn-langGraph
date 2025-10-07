
## Streaming in LangGraph with Ollama

Generating full responses from LLMs can take several seconds, especially in complex workflows with multiple model calls. To improve user experience, streaming allows you to display intermediate outputs as they are generated, rather than waiting for the entire response.

Both LangChain and LangGraph provide built-in support for streaming, making them ideal for real-time applications like chatbots, dashboards, and interactive UIs.

Why Streaming?

- See responses as they are generated (token-by-token or chunk-by-chunk)
- Useful for chat, progress bars, and monitoring workflows

```python
from langgraph.graph import StateGraph
from langgraph.llms import Ollama

# Define state schema
class MyState(dict):
    pass

# Create Ollama LLM node (Ollama must be running locally)
llm = Ollama(model="llama2", streaming=True)

def generate_node(state: MyState):
    question = state["question"]
    # Streaming response from Ollama
    response = llm.invoke(question)
    return {"answer": response.content}

# Build and compile graph
graph = StateGraph(MyState)
graph.add_node("generate", generate_node)
graph.set_entry_point("generate")
compiled_graph = graph.compile()

# Stream output from the graph
input_message = "What is Task Decomposition?"
for message, metadata in compiled_graph.stream(
    {"question": input_message},
    stream_mode="messages",
):
    if metadata["langgraph_node"] == "generate":
        print(message.content, end="|")
```

Notes:
- Ollama must be installed and running locally with the desired model (e.g., `llama2`).
- `stream_mode="messages"` streams token-by-token output from the chat model node.

### Streaming Methods

- astream / stream → high‑level streaming of outputs or state from a Runnable (incl. compiled LangGraph graphs). Useful for quickly getting results as they’re produced, without diving into every internal.

| Mode       | Description                                                      |
|------------|------------------------------------------------------------------|
| `values`   | Streams all state values at each step.                          |
| `updates`  | Streams node names and updates after each step.                 |
| `debug`    | Streams debug events for each step.                             |
| `messages` | Streams LLM messages token-by-token (ideal for chat models).    |
| `custom`   | Streams custom output using LangGraph's `StreamWriter`.         |


> Tip: For LangGraph graphs, you can stream messages / updates / values (three modes) to control how much you receive at each step. This is helpful if you want just the new message, only diffs, or full state snapshots as the graph evolves. 

- astream_events → low‑level streaming of event objects (typed as StreamEvent) for every step in your run (LLM tokens, tool start/end, chain start/end, etc.). Ideal for UIs and telemetry where you need tokens and lifecycle hooks. 

> Token‑level streaming depends on whether your model/provider supports streaming (and you enable it). Otherwise you’ll get message‑level updates when the model finishes.

