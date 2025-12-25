import asyncio
from typing import TypedDict, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_mcp_adapters.client import MultiServerMCPClient

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]

# Configure MultiServerMCPClient to connect to multiple tool servers.
# Each server is started as a subprocess using stdio for communication.
# The dictionary keys are server names used for tool loading.
client = MultiServerMCPClient({
    "mongodb": {
        "command": "python",
        "args": ["graph_mcp/servers/mongodb_server.py"],
        "transport": "stdio",
    },
    "openapi": {
        "command": "python",
        "args": ["graph_mcp/servers/api_server.py"],
        "transport": "stdio",
    },
    "internal": {
        "command": "python",
        "args": ["graph_mcp/servers/internal_server.py"],
        "transport": "stdio",
    },
})


async def main():
    # List of servers to load tools from
    server_names = ["openapi", "internal", "mongodb"]
    tools = []

    # Load tools from each server with error handling
    for name in server_names:
        try:
            server_tools = await client.get_tools(server_name=name)
            tools.extend(server_tools)
            print(f"✅ Loaded tools from '{name}': {[t.name for t in server_tools]}")
        except Exception as e:
            print(f"❌ Failed to load tools from '{name}': {e}")

    # Exit if no tools were loaded
    if not tools:
        print("🚨 No tools loaded. Exiting.")
        return

    llm = ChatOllama(model="llama3.2:3b", temperature=0, streaming=True).bind_tools(tools)

    # LLM Node
    async def llm_processor_node(state: AgentState) -> AgentState:
        system_prompt = SystemMessage(
            content="You are an AI assistant. Use tools for math, weather, or user data queries."
        )
        response = await llm.ainvoke([system_prompt] + state["messages"])
        state["messages"].append(response)
        return state

    async def should_continue_node(state: AgentState) -> str:
        last_message = state["messages"][-1]
        return "continue" if getattr(last_message, "tool_calls", None) else "end"


    graph = StateGraph(AgentState)
    graph.add_node("llm_processor_node", llm_processor_node)
    graph.add_node("tool_nodes", ToolNode(tools=tools))

    # Define graph edges and flow
    graph.add_edge(START, "llm_processor_node")
    graph.add_conditional_edges("llm_processor_node", should_continue_node, {
        "continue": "tool_nodes",
        "end": END
    })
    graph.add_edge("tool_nodes", "llm_processor_node")

    # Compile the graph into an executable app
    app = graph.compile()


    print("💬 Type 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        # Initialize conversation state with user input
        input_state = {"messages": [HumanMessage(content=user_input)]}
        print("📡 Streaming events...\n")

        # Stream events from the graph execution
        try:
            async for event in app.astream_events(input=input_state, version="v2"):
                kind = event["event"]
                name = event.get("name")

                # Handle different event types
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk and chunk.content:
                        print(chunk.content, end="", flush=True)
                elif kind == "on_tool_start":
                    tool_input = event["data"].get("input")
                    print(f"\n🛠️ Tool start: {name} | input: {tool_input}")
                elif kind == "on_tool_end":
                    tool_output = event["data"].get("output")
                    print(f"\n✅ Tool end: {name} | output: {tool_output}")
                elif kind == "on_chain_end":
                    print()
        except Exception as e:
            print(f"⚠️ Error during streaming: {e}")

# Run the async main function using asyncio
if __name__ == "__main__":
    asyncio.run(main())
