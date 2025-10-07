"""
LangGraph Streaming Demo

- ToolNode: Executes @tool functions and returns ToolMessages.
- add_messages reducer: Appends new messages to state.
- Stream modes: "messages", "updates", "values" for different streaming behaviors.

References:
- https://github.com/langchain-ai/langgraph
- https://python.langchain.com/docs/versions/v0_2/migrating_astream_events/
"""

from typing import TypedDict, Sequence, Annotated
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
import requests

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def get_weather(city: str) -> str:
    """Get the current temperature in Celsius for a given city."""
    try:
        geo_resp = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}")
        geo = geo_resp.json()
        if not geo.get("results"):
            return f"City '{city}' not found."
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]
        weather_resp = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        )
        weather = weather_resp.json()
        temp = weather["current_weather"]["temperature"]
        return f"The current temperature in {city} is {temp}°C."
    except Exception as e:
        return f"Error fetching weather: {e}"

tool_list = [get_weather]
llm = ChatOllama(model="llama3.2:3b", temperature=0).bind_tools(tool_list)

def llm_processor_node(state: AgentState) -> AgentState:
    """Process messages with LLM and append response."""
    system_prompt = SystemMessage(
        content="You are an AI assistant. Use tools for math, weather, or user data queries."
    )
    response = llm.invoke([system_prompt] + list(state["messages"]))
    return {"messages": [response]}

def should_continue_node(state: AgentState) -> str:
    """Decide whether to continue or end based on tool calls."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"

graph = StateGraph(AgentState)
graph.add_node("llm_processor_node", llm_processor_node)
tool_nodes = ToolNode(tools=tool_list)
graph.add_node("tool_nodes", tool_nodes)
graph.add_edge(START, "llm_processor_node")
graph.add_conditional_edges("llm_processor_node", should_continue_node, {
    "continue": "tool_nodes",
    "end": END
})
graph.add_edge("tool_nodes", "llm_processor_node")
app = graph.compile()

# Streaming modes
def stream_messages(user_text: str):
    state_in: AgentState = {"messages": [HumanMessage(content=user_text)]}
    print("⏳ Processing...")
    for message, meta in app.stream(input=state_in, stream_mode="messages"):
        print(message.content, end="", flush=True)
    print()

def stream_updates(user_text: str):
    state_in: AgentState = {"messages": [HumanMessage(content=user_text)]}
    print("🔄 Streaming updates...")
    for update in app.stream(input=state_in, stream_mode="updates"):
        for node_name, delta in update.items():
            print(f"[{node_name}] {list(delta.keys())} updated")
            if "messages" in delta:
                msg = delta["messages"][-1]
                print("  ↳", getattr(msg, "content", msg))

def stream_values(user_text: str):
    state_in: AgentState = {"messages": [HumanMessage(content=user_text)]}
    print("🧩 Streaming full state values...")
    final_state = None
    for state in app.stream(input=state_in, stream_mode="values"):
        final_state = state
        last = state["messages"][-1]
        print("State changed → last message:", getattr(last, "content", last))
    print("\n✅ Done. Final assistant message:", final_state["messages"][-1].content)

# Interactive loop (default: messages)
if __name__ == "__main__":
    while True:
        user_input = input("\n👤 You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting...")
            break
        # stream_messages(user_input)
        stream_updates(user_input)