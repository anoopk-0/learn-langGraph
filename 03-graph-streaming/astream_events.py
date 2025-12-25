"""
Async event streaming demo with LangGraph and ToolNode.

- Async event streaming: on_chat_model_stream, on_tool_start, on_tool_end, on_chain_end.

"""
import asyncio
import requests
from typing import TypedDict, Sequence, Annotated
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def get_weather(city: str) -> str:
    """Get the current temperature in Celsius for a given city via Open-Meteo."""
    try:
        geo_resp = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}",
            timeout=10,
        )
        geo_resp.raise_for_status()
        geo = geo_resp.json()
        if not geo.get("results"):
            return f"City '{city}' not found."

        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]

        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": "true"},
            timeout=10,
        )
        weather_resp.raise_for_status()
        weather = weather_resp.json()
        temp = weather["current_weather"]["temperature"]
        return f"The current temperature in {city} is {temp}°C."
    except Exception as e:
        return f"Error fetching weather: {e}"


tool_list = [get_weather]

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    streaming=True,
).bind_tools(tool_list)


def llm_processor_node(state: AgentState) -> AgentState:
    """Call the LLM with the conversation so far and append AI response."""
    system_prompt = SystemMessage(
        content=(
            "You are an AI assistant. When the user asks about weather, call the tool. "
            "Otherwise, answer directly. Keep responses concise."
        )
    )
    response = llm.invoke([system_prompt] + list(state["messages"]))
    # Return a partial update; `add_messages` will append it to state
    return {"messages": [response]}


def should_continue_node(state: AgentState) -> str:
    """Route to ToolNode if the model requested tools; otherwise end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"


graph = StateGraph(AgentState)
graph.add_node("llm_processor_node", llm_processor_node)
tool_nodes = ToolNode(tools=tool_list)
graph.add_node("tool_nodes", tool_nodes)

graph.add_edge(START, "llm_processor_node")
graph.add_conditional_edges(
    "llm_processor_node",
    should_continue_node,
    {"continue": "tool_nodes", "end": END},
)
graph.add_edge("tool_nodes", "llm_processor_node")

app = graph.compile()


async def demo_events(user_text: str):
    input_state = {"messages": [HumanMessage(content=user_text)]}
    print("📡 Event streaming (v2)...\n")

    async for event in app.astream_events(input=input_state, version="v2"):
        kind = event["event"]
        name = event.get("name")  # often the node or runnable name

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk and chunk.content:
                print(chunk.content, end="", flush=True)

        elif kind == "on_tool_start":
            tool_input = event["data"].get("input")
            print(f"\n🛠️  Tool start: {name} | input: {tool_input}")

        elif kind == "on_tool_end":
            tool_output = event["data"].get("output")
            print(f"\n✅ Tool end: {name} | output: {tool_output}")

        elif kind == "on_chain_end":
            print()


if __name__ == "__main__":
    async def main():
        print("Type 'quit' to exit.")
        while True:
            user_text = input("\n👤 You: ").strip()
            if user_text.lower() in {"quit", "exit"}:
                print("Bye!")
                break
            await demo_events(user_text)

    asyncio.run(main())
