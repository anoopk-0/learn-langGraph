## What is MCP?

MCP (Model Context Protocol) is an open-source standard for connecting AI applications to external systems.

MCP servers wrap your business logic, database queries, or external API calls and make them accessible to agents in a standardized way.

> LangGraph agents can use tools defined on MCP servers through the "langchain-mcp-adapters" library.

- Expose database queries (MongoDB, SQL) as tools for agents.
- Wrap external APIs (weather, user info) for easy agent access.
- Provide internal business logic (math, greeting, calculations) as callable tools.

There are several ways agents and clients can interact with MCP servers:

- Synchronous (Most Common):  
  The client sends a request and waits for the server to process and return the response. This is the typical pattern for API calls and is widely used because it is simple and predictable. For example, when an agent calls a tool to fetch user data, it pauses until the result is received.

- Asynchronous:  
  The client sends a request and continues other work without waiting for the response. The server processes the request and delivers the result later, often via a callback or notification. This is useful for long-running tasks or when immediate results are not required.

- Request-Response:  
  A direct interaction where the client asks the agent to perform an action ("Do this") and receives the outcome ("Here is the result"). This pattern is foundational to most API and tool interactions.

- Event-Driven:  
  Agents or tools emit events when something happens (e.g., data updated, error occurred). Other agents or clients can listen for these events and react accordingly. This enables reactive workflows and decouples components.

- Batch Operations:  
  The client can send multiple requests together in a single batch. The server processes all requests and returns the results collectively. This improves efficiency when handling large volumes of similar tasks.


## Why is MCP Needed?

Modern AI agents often need to interact with external systems—databases, APIs, business logic, and other services. Each system may have its own interface, protocol, and data format, making integration complex and error-prone.

MCP solves this by:
- Providing a standardized way for agents to call external tools and services, regardless of their underlying technology.
- Enabling interoperability between agents and tools written in different languages or frameworks.
- Simplifying the process of exposing new capabilities to agents, so developers can focus on business logic rather than integration details.
- Supporting scalable, modular architectures where tools and agents can be added, removed, or updated independently.

With MCP, you can build AI workflows that are flexible, maintainable, and easy to extend—making it a foundational protocol for modern agent-based systems.

### Tool Handling

- Agents use tools to perform tasks. You can add new tools without changing the main code.
- Each tool has a clear interface (inputs and outputs).

### JSON-RPC Communication

Agents and clients communicate using JSON-RPC, a lightweight protocol for remote procedure calls encoded in JSON. JSON-RPC works by sending structured messages between a client and server, where each message specifies the method to invoke, any parameters, and a unique identifier for tracking responses.

How JSON-RPC Communication Works:

1. Request:  
  The client sends a JSON object containing:
  - `"jsonrpc"`: The protocol version (usually `"2.0"`).
  - `"method"`: The name of the function or tool to call.
  - `"params"`: The input parameters for the method (can be an object or array).
  - `"id"`: A unique identifier to match the response with the request.

2. Processing:  
  The server receives the request, locates the specified method, and executes it using the provided parameters.

3. Response:  
  The server sends back a JSON object containing:
  - `"jsonrpc"`: The protocol version.
  - `"result"`: The output of the method (if successful).
  - `"error"`: An error object (if something went wrong).
  - `"id"`: The same identifier as in the request, so the client can match responses to requests.

4. Interoperability:  
  Because JSON-RPC uses plain JSON, it is language-agnostic. Any client or server that understands JSON can participate, making it easy to connect systems written in different programming languages.

This communication pattern allows agents and clients to interact in a standardized, predictable way, supporting both synchronous and asynchronous workflows.

A typical JSON-RPC request looks like:

```json
{
  "jsonrpc": "2.0",
  "method": "greet_user",
  "params": { "name": "Alice" },
  "id": 1
}
```

And the response:

```json
{
  "jsonrpc": "2.0",
  "result": "Hello, Alice!",
  "id": 1
}
```

## Architecture Overview

MCP enables AI applications to interact with external systems in a standardized way. MCP focuses solely on the protocol for context exchange and does not prescribe how AI applications use LLMs or manage context.

Each MCP client maintains a one-to-one connection with its corresponding MCP server, allowing the host to interact with multiple servers simultaneously (local or remote).

MCP uses a client-server architecture:

- MCP Host: The AI application (e.g., Claude Code, Visual Studio Code) that manages one or more MCP clients.
- MCP Client: Maintains a dedicated connection to an MCP server and fetches context for the host.
- MCP Server: Provides context, tools, resources, and prompts to MCP clients.

MCP is structured into two distinct layers, each responsible for different aspects of communication and integration:

#### 1. Data Layer

The Data Layer specifies the format and semantics of messages exchanged between MCP clients and servers. It is built around the JSON-RPC protocol, which provides a standardized way to perform remote procedure calls using JSON. This layer defines:

- Message Structure: All requests and responses follow the JSON-RPC specification, including fields for protocol version, method name, parameters, results, errors, and unique identifiers.
- Lifecycle Management: Handles the creation, execution, and termination of tool invocations, resource access, and prompt exchanges.
- Core Primitives: Establishes the basic building blocks for interaction, such as:
  - Tools: Functions or operations exposed by the server that agents can invoke.
  - Resources: Data or services made available to clients.
  - Prompts: Contextual information or instructions for agents.
  - Notifications: Event messages sent from server to client to signal changes or updates.

This layer ensures that all MCP interactions are consistent, predictable, and language-agnostic, enabling interoperability across diverse systems.

#### 2. Transport Layer

The Transport Layer is responsible for delivering messages between clients and servers. It abstracts the underlying communication mechanism, allowing MCP to operate over different channels. Key responsibilities include:

- Channel Management: Supports multiple transport options, such as:
  - Stdio: For local communication, using standard input/output streams.
  - HTTP: For remote communication over the web.
  - Other Protocols: Potential support for WebSockets, TCP, or custom transports.
- Connection Establishment: Handles the setup and teardown of connections, ensuring reliable message delivery.
- Message Framing: Packages messages for transmission, managing boundaries and encoding to prevent data loss or corruption.
- Authentication: Provides mechanisms for verifying client and server identities, securing communication against unauthorized access.

By separating data handling from transport concerns, MCP allows developers to choose the most appropriate communication channel for their environment, while maintaining a consistent protocol for message exchange.


### Example: Exposing a Tool with LangGraph
Here’s a simple example using LangGraph to expose a greeting tool via MCP:

```python
from langgraph.graph import Graph
from langgraph.tools import Tool

# Define the greeting function
def greet_user(name: str) -> str:
  return f"Hello, {name}!"

# Wrap the function as a Tool
greet_tool = Tool(
  name="greet_user",
  description="Greets the user by name.",
  func=greet_user,
  input_schema={"name": str},
  output_schema=str
)

# Create a graph and register the tool
graph = Graph()
graph.add_tool(greet_tool)

# Simulate a JSON-RPC request
request = {
  "jsonrpc": "2.0",
  "method": "greet_user",
  "params": {"name": "Alice"},
  "id": 1
}

# Process the request and generate a response
result = graph.call_tool(request["method"], request["params"])
response = {
  "jsonrpc": "2.0",
  "result": result,
  "id": request["id"]
}

print(response)  # {'jsonrpc': '2.0', 'result': 'Hello, Alice!', 'id': 1}
```

```txt
Client ----> Agent ----> Tool
   |           |           |
   |           |           |
   +-----------+-----------+

- The client sends a request to the agent.
- The agent decides which tool to use and sends the job to that tool.
- The tool does the work and sends the result back to the agent, which returns it to the client.

You can have many agents and tools, all connected like a web, working together.
```
