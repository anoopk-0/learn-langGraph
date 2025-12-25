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

# Wrap functions as FastMCP tools
fastmcp_tools = [to_fastmcp(multiply), to_fastmcp(greet)]

# Create and start the MCP server with stdio transport (local communication)
mcp = FastMCP("InternalTools", tools=fastmcp_tools)

# Using "stdio" transport is recommended for local communication, especially when integrating with subprocesses.
# It allows the server to communicate via standard input/output, which is simple and efficient for local tools.
mcp.run(transport="stdio")
