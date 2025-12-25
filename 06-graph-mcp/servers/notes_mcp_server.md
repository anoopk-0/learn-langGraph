## What is an MCP Server?

An MCP (Model Context Protocol) server is a service that exposes Python functions, APIs, or external services as callable tools for AI agents. The server acts as a bridge between agents and your business logic, making it easy for agents to invoke tools using a standardized protocol (typically JSON-RPC).

MCP servers can be implemented using various frameworks. One popular implementation is FastMCP, which provides a simple way to register tools and expose them over different transports (e.g., stdio, TCP).

- **Tool Registration:** Each function is wrapped and registered as a tool. The MCP server knows how to call each tool and what inputs/outputs to expect.
- **Transport:** The server can communicate over different transports. `stdio` is simple and local; `tcp` allows remote access.
- **Agent Interaction:** Agents send JSON-RPC requests to the server, specifying the tool and parameters. The server executes the tool and returns the result.

### Choosing a Transport

FastMCP supports multiple transports for agent-server communication. The two most common options are:

#### 1. stdio Transport

**stdio**: Communicates using standard input and output streams. Best suited for local development, quick testing, or scenarios where both the agent and server run on the same machine or process. This method is straightforward, fast, and requires no network configuration.

```python
mcp = FastMCP("OpenAPI", tools=fastmcp_tools)
mcp.run(transport="stdio")
```

#### 2. tcp Transport

**tcp**: Uses a TCP socket to allow agents or clients to connect to the server over the network. Specify the `host` and `port` to control where the server listens. Choose TCP for remote access, connecting agents and servers across machines, or production environments where distributed or multi-agent setups are needed.

```python
mcp = FastMCP("OpenAPI", tools=fastmcp_tools)
mcp.run(transport="tcp", host="127.0.0.1", port=8000)
```

### Example

Let's walk through how to expose Python functions as MCP tools and run the server.

#### 1. Define Tool Functions

Create the Python functions you want to expose:

```python
def call_openapi(query: str) -> dict:
    # Call an OpenAPI endpoint
    return {"result": f"OpenAPI response for {query}"}

def get_weather(city: str) -> dict:
    # Fetch weather data
    return {"city": city, "weather": "sunny"}

def get_users() -> list:
    # Fetch user list
    return ["Alice", "Bob", "Charlie"]
```

#### 2. Register Functions as MCP Tools

Wrap your functions using `to_fastmcp` so they can be registered with the server:

```python
from fastmcp import to_fastmcp

fastmcp_tools = [
    to_fastmcp(call_openapi),
    to_fastmcp(get_weather),
    to_fastmcp(get_users)
]
```

#### 3. Create and Run the MCP Server

Instantiate the server and choose a transport:

```python
from fastmcp import FastMCP

mcp = FastMCP("OpenAPI", tools=fastmcp_tools)

# Run with stdio transport (local)
mcp.run(transport="stdio")

# Or run with tcp transport (networked)
mcp.run(transport="tcp", host="127.0.0.1", port=8000)
```

This setup allows agents to call your Python functions as tools via JSON-RPC, using either local or remote communication.
