from typing import TypedDict, Sequence, Annotated
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import requests

class AgentState(TypedDict):
    """Agent state containing a list of messages exchanged during the conversation."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Tool Definitions
@tool
def add(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b

@tool
def get_weather(city: str) -> str:
    """Get the current temperature in Celsius for a given city using Open-Meteo API."""
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
    try:
        users_resp = requests.get("https://jsonplaceholder.typicode.com/users")
        users = users_resp.json()
        return "\n".join([f"{u['id']}: {u['name']} ({u['email']})" for u in users[:10]])
    except Exception as e:
        return f"Error fetching users: {e}"


# Tool Binding
tool_list = [add, get_weather, get_users]
llm = ChatOllama(model="llama3.2:3b", temperature=0).bind_tools(tool_list)
tools_dict = {tool.name: tool for tool in tool_list}


"""
Custom tool execution node.
This manually handles tool calls from the LLM's response and returns ToolMessages.

🔍 Difference from ToolNode:
- ToolNode (from langgraph.prebuilt) is a generic handler that automatically executes tools.
- This custom node gives you full control over:
    - Error handling
    - Argument parsing
    - Logging
    - Custom tool routing
"""
# Custom Tool Execution Node
def tool_node(state: AgentState) -> AgentState:
    """
    Executes tool calls from the LLM's response and returns results as ToolMessages.
    Handles missing tools and execution errors gracefully.
    """
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        tool_name = t['name']
        args = t.get('args', {})
        query = args.get('query', '')  # fallback for tools expecting 'query'
        if tool_name not in tools_dict:
            result = f"Tool '{tool_name}' does not exist."
        else:
            try:
                result = tools_dict[tool_name].invoke(query)
            except Exception as e:
                result = f"Error executing tool '{tool_name}': {e}"
        results.append(ToolMessage(tool_call_id=t['id'], name=tool_name, content=str(result)))
    return {'messages': results}


# LLM Processor Node
def llm_processor_node(state: AgentState) -> AgentState:
    """Processes messages with the LLM and appends the response to the state."""
    system_prompt = SystemMessage(
        content="You are an AI assistant. Use tools for math, weather, or user data queries."
    )
    response = llm.invoke([system_prompt] + state["messages"])
    state["messages"].append(response)
    return state


# Conditional Node
def should_continue_node(state: AgentState) -> str:
    """Decides whether to continue with tool execution or end the graph."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"


# Graph Construction
graph = StateGraph(AgentState)
graph.add_node("llm_processor_node", llm_processor_node)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "llm_processor_node")
graph.add_conditional_edges("llm_processor_node", should_continue_node, {
    "continue": "tool_node",
    "end": END
})
graph.add_edge("tool_node", "llm_processor_node")


# Compile the Agent
app = graph.compile()


# Stream Output Helper
def print_stream(stream):
    """Prints each message from the agent stream."""
    for s in stream:
        message = s["messages"][-1]
        print(getattr(message, "content", message))


# Entry Point
if __name__ == "__main__":
    inputs = {"messages": [HumanMessage(content="let list of 5 user in a table format")]}
    print_stream(app.stream(inputs, stream_mode="values"))
