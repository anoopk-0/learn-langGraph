"""
ToolNode Example (LangGraph)
----------------------------

Demonstrates how to use ToolNode in LangGraph to enable an agent to call tools for tasks such as math, weather, and user data queries.
ToolNode manages tool execution, error handling, and message passing in the graph.
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
    """Agent state containing a list of messages."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    # Example tool: simple math operation
    return a + b

@tool
def get_weather(city: str) -> str:
    """Get the current temperature in Celsius for a given city."""
    # Example tool: fetches weather data from an external API
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

@tool
def get_users() -> str:
    """Fetch 10 users from JSONPlaceholder API."""
    # Example tool: fetches user data from an external API
    try:
        users_resp = requests.get("https://jsonplaceholder.typicode.com/users")
        users = users_resp.json()
        return "\n".join([f"{u['id']}: {u['name']} ({u['email']})" for u in users[:10]])
    except Exception as e:
        return f"Error fetching users: {e}"


# List of all available tools for the agent
tool_list = [add, get_weather, get_users]

# Bind tools to the LLM so it can call them when needed
llm = ChatOllama(model="llama3.2:3b", temperature=0).bind_tools(tool_list)

# Node that processes messages with the LLM and appends the response
def llm_processor_node(state: AgentState) -> AgentState:
    """Process messages with LLM and append the response to state."""
    system_prompt = SystemMessage(
        content="You are an AI assistant. Use tools for math, weather, or user data queries."
    )
    response = llm.invoke([system_prompt] + state["messages"])
    state["messages"].append(response)
    return state


# Node that decides whether to continue (call a tool) or end
def should_continue_node(state: AgentState) -> str:
    """Return 'continue' if the last message requests a tool, else 'end'."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("llm_processor_node", llm_processor_node)

"""
# NOTE: ToolNode executes registered tools and returns ToolMessages to the agent state
ToolNode is a prebuilt node that manages tool execution in the graph.

It automatically detects tool calls from the agent, executes the requested tools, and returns ToolMessages containing the results back to the agent state. The decision of which tool to call is managed by the LLM (Large Language Model). The LLM analyzes the user's input and, if it determines that a tool is needed (e.g., for math, weather, or data lookup), it outputs a tool call. The agent then routes this request to the ToolNode, which executes the specified tool and returns the result.

This enables the agent to interact with external functions (tools) seamlessly, handling tool invocation, error reporting, and message passing in a single node.

For ToolNode:

    - ToolNode is only activated when the graph's logic (conditional edges) routes execution to it.
"""
tool_nodes = ToolNode(tools=tool_list)
graph.add_node("tool_nodes", tool_nodes)

graph.add_edge(START, "llm_processor_node")
graph.add_conditional_edges("llm_processor_node", should_continue_node, {
    "continue": "tool_nodes",
    "end": END
})
graph.add_edge("tool_nodes", "llm_processor_node")

# Compile the graph into an executable agent
app = graph.compile()

def print_stream(stream):
    """Prints each message from the agent stream."""
    for s in stream:
        message = s["messages"][-1]
        print(getattr(message, "content", message))

if __name__ == "__main__":
    inputs = {"messages": [HumanMessage(content="Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
    print_stream(app.stream(inputs, stream_mode="values"))