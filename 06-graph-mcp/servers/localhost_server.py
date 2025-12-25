# MCP server exposing internal tools over TCP on localhost:9000.
# Agents can connect using transport="tcp" with the specified host and port.

from langchain_core.tools import tool
from langchain_mcp_adapters.tools import to_fastmcp
from mcp.server.fastmcp import FastMCP

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers and return the result."""
    return a * b

@tool
def greet(name: str) -> str:
    """Return a greeting for the given name."""
    return f"Hello, {name}!"

# Wrap functions as FastMCP tools for remote invocation
fastmcp_tools = [to_fastmcp(multiply), to_fastmcp(greet)]

# Create the MCP server and register the tools
mcp = FastMCP("InternalTools", tools=fastmcp_tools)

# Run the MCP server using TCP transport on localhost:9000
# Use TCP when you want remote agents or processes to connect over the network,
# or when you need multiple clients to access the server concurrently.
# Use stdio for local, single-process communication (e.g., when embedding the server in a subprocess).
mcp.run(transport="tcp", host="127.0.0.1", port=9000)
