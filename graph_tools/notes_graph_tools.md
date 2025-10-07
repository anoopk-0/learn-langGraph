## LangGraph Tools: Detailed Notes

Tools in LangGraph are callable functions or modules that can be invoked by nodes in a graph to perform specific tasks, such as fetching data, performing calculations, or interacting with external APIs. They enable your graph-based agent to extend its capabilities beyond simple message passing.

- ToolNode: A special node in LangGraph designed to execute tools. It handles tool calls, manages parallel execution, error handling, and returns results as ToolMessages.
- @tool Decorator: Used to mark a Python function as a tool, making it discoverable and callable by the graph.
- ToolMessage: The message type returned by a tool after execution, which is appended to the agent's state.
- Tool List: A collection of tool functions registered with the graph or LLM, allowing dynamic selection and invocation.

### How Tools Work in LangGraph

1. Definition: Tools are defined as Python functions, decorated with `@tool`. Each tool should have a clear docstring describing its purpose and expected input/output.

2. Registration: Tools are registered with the graph or LLM using a tool list. This enables the agent to call them when needed.

3. Invocation: When the agent determines a tool is needed (e.g., based on user input or LLM output), it routes the request to the ToolNode, which executes the tool and returns the result.

4. Result Handling: The output of the tool is wrapped in a ToolMessage and added to the agent's state, allowing further processing or response to the user.

```python
from langchain_core.tools import tool

@tool
def add_user_to_db(name: str, email: str) -> str:
    # Code to insert user into DB
    return "User added."

@tool
def get_user_from_db(user_id: int) -> dict:
    # Code to fetch user from DB
    return {"id": user_id, "name": "John Doe"}

@tool
def get_weather(city: str) -> str:
	"""Get the current temperature in Celsius for a given city."""
	# ...implementation...


- Write clear docstrings for each tool, describing its function and expected arguments.
- Handle errors gracefully within tools to avoid breaking the graph execution.
- Use ToolNode for managing multiple tools and parallel execution.
- Keep tools stateless and idempotent when possible.

```

#### Common Use Cases

- Fetching weather, stock prices, or other external data.
- Performing calculations or data transformations.
- Interacting with databases or APIs.
- Automating repetitive tasks within a workflow.

### References

- [LangGraph Documentation](https://github.com/langchain-ai/langgraph)
- [LangChain Tools](https://python.langchain.com/docs/modules/agents/tools/)

