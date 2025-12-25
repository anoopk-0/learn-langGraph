import logging
from typing import List, TypedDict, Any, Dict, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import SummarizationNode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: List[BaseMessage]

def get_llm(model_name: str = "llama3.2:3b", temperature: float = 0) -> ChatOllama:
    """Factory for LLM instance."""
    return ChatOllama(model=model_name, temperature=temperature)


# Summarizer builder
# NOTE: Keep output_messages_key as-is (or set to "messages" if supported).
def build_summarization_node(llm: ChatOllama) -> SummarizationNode:
    """
    Builds the summarization node.

    If SummarizationNode supports `input_messages_key`/`output_messages_key`,
    you could set output_messages_key="messages" to avoid adapting later.
    This example keeps `llm_input_messages` to demonstrate safe adaptation.
    """
    return SummarizationNode(
        token_counter=count_tokens_approximately,
        model=llm,
        max_tokens=384,
        max_summary_tokens=128,
        output_messages_key="llm_input_messages",
        # If available in your version, you can also set:
        # input_messages_key="messages",
    )


# Summarization adapter node
# Ensures the graph state remains { "messages": [...] }.
def summarization_adapter_node(summarizer: SummarizationNode):
    """
    Wraps SummarizationNode so the graph always carries `messages` in state.

    Behavior:
      - If no messages in state -> pass-through (no summarization).
      - Else -> run summarizer; then map its output back to state["messages"].
    """
    def _node(state: AgentState) -> AgentState:
        messages: List[BaseMessage] = state.get("messages", [])
        if not messages:
            logger.info("No messages in state; skipping summarization and passing through.")
            return state

        # Call the summarizer with the state. Depending on the SummarizationNode contract,
        # it may look at `messages` by default or need an input key (already provided above).
        try:
            out: Dict[str, Any] = summarizer.invoke(state)
        except Exception as e:
            logger.error(f"SummarizationNode failed: {e}. Passing through original messages.")
            return state

        # Prefer the configured output key, fallback to "messages" or original messages.
        trimmed: Optional[List[BaseMessage]] = (
            out.get("llm_input_messages")
            or out.get("messages")
            or messages
        )

        # Return state with normalized key
        return AgentState(messages=trimmed)

    return _node


def llm_node_factory(llm: ChatOllama):
    def llm_node(state: AgentState) -> AgentState:
        """LLM node for graph execution."""
        try:
            response = llm.invoke(state["messages"])
            state["messages"].append(response)
            logger.info("LLM response appended to state.")
        except Exception as e:
            logger.error(f"Error in llm_node: {e}")
            raise
        return state
    return llm_node


# Build the graph
def build_graph(llm: ChatOllama) -> StateGraph:
    """Builds the agent graph."""
    graph = StateGraph(AgentState)

    summarizer = build_summarization_node(llm)
    graph.add_node("summarization", summarization_adapter_node(summarizer))
    graph.add_node("llm_node", llm_node_factory(llm))

    graph.add_edge(START, "summarization")
    graph.add_edge("summarization", "llm_node")
    graph.add_edge("llm_node", END)

    return graph


# Stream printer

def print_stream(stream: Any) -> None:
    """
    Prints updates from the agent stream.

    All nodes in this graph now maintain `messages` in their state,
    thanks to the adapter. If you ever output alternative keys, add
    a small fallback here (shown in comment).
    """
    for chunk in stream:
        for node, update in chunk.items():
            logger.info(f"Update from node: {node}")
            # Fallback example if you later change output keys:
            # msgs = update.get("messages") or update.get("llm_input_messages") or []
            msgs = update.get("messages", [])
            for message in msgs:
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()
        print("\n")


# Main

def main() -> None:
    """Main entrypoint for running the agent interactively."""
    config: Dict[str, Any] = {"configurable": {"thread_id": "1"}}
    llm_instance = get_llm()
    graph = build_graph(llm_instance)
    app = graph.compile(checkpointer=InMemorySaver())

    print("🤖 LangGraph Agent is running. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in {"exit", "quit"}:
                print("👋 Exiting agent.")
                break
            inputs = AgentState(messages=[HumanMessage(content=user_input)])
            print_stream(app.stream(inputs, config=config, stream_mode="updates"))
        except KeyboardInterrupt:
            print("\n👋 Exiting agent.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
