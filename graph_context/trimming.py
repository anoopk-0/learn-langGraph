from typing import List, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class AgentState(TypedDict):
    messages: List[BaseMessage]



# NOTE:
# The function uses `trim_messages` to reduce the message history based on token count.
#     - Trimming strategy is set to "last", meaning the most recent messages are kept.
#     - Only messages between "human" and "tool" roles are considered for trimming.
#     - The `count_tokens_approximately` function is used to estimate token usage.
#     - The maximum allowed tokens after trimming is 384.
def pre_model_hook(state: AgentState) -> dict:
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=384,
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"messages": trimmed_messages}

llm = ChatOllama(model="llama3.2:3b", temperature=0)

def trim_messages_node(state: AgentState) -> AgentState:
    return pre_model_hook(state)

def llm_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(response)
    return state

graph = StateGraph(AgentState)

graph.add_node("trim_messages", trim_messages_node)
graph.add_node("llm_node", llm_node)

graph.add_edge(START, "trim_messages")
graph.add_edge("trim_messages", "llm_node")
graph.add_edge("llm_node", END)

app = graph.compile(checkpointer=InMemorySaver())

def print_stream(stream):
    for chunk in stream:
        for node, update in chunk.items():
            print(f"Update from node: {node}")
            for message in update.get("messages", []):
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()
        print("\n\n")

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    print("🤖 LangGraph Agent is running. Type 'exit' or 'quit' to stop.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Exiting agent.")
            break
        inputs = AgentState(messages=[HumanMessage(content=user_input)])
        print_stream(app.stream(inputs, config=config, stream_mode="updates"))
