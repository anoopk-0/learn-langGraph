"""
LangGraph enables you to update graph state and control flow simultaneously by returning a `Command` object from node functions.
This approach allows you to specify both the next node and any state changes in a single return value.

    - update: Dict of state changes to apply immediately.
    - goto: Name(s) of the next node(s) to execute (string, list, or Send objects).
    - resume: Mapping of interrupt_id → value to resume execution after an interrupt.
    - graph: Which graph to execute next (e.g., Command.PARENT for parent graph handoff).
"""

from typing import Literal
from langgraph.types import Command

"""
NOTE:
When returning `Command`, always add a return type annotation with the list of node names the node can route to, e.g. `Command[Literal["my_other_node"]]`. This is required for graph rendering and tells LangGraph about possible navigation paths.
"""

# Example 1: Basic Usage
def my_node(state) -> Command[Literal["my_other_node"]]:
    """
    Updates the state and routes to 'my_other_node'.
    """
    return Command(
        update={"foo": "bar"},   # Update the state dictionary
        goto="my_other_node"     # Specify the next node to execute
    )

# Example 2: Dynamic Control Flow
def my_node(state) -> Command[Literal["my_other_node"]]:
    """
    Conditionally updates the state and routes based on current state.
    """
    if state.get("foo") == "bar":
        # If 'foo' is 'bar', update it and route to 'my_other_node'
        return Command(update={"foo": "baz"}, goto="my_other_node")
    # Optionally, handle other cases or return a default Command if needed

"""
NOTE: When to Use Command vs. Conditional Edges
Use `Command` when you need to both update the graph state and route to a different node (e.g., multi-agent handoffs).
Use conditional edges to route between nodes conditionally **without** updating the state.
"""

## Navigating to a Node in a Parent Graph (Subgraphs)
# If you are using subgraphs, you can navigate from a node within a subgraph to a node in the parent graph by specifying `graph=Command.PARENT`:

def my_node(state) -> Command[Literal["other_subgraph"]]:
    return Command(
        update={"foo": "bar"},
        goto="other_subgraph",  # Node in the parent graph
        graph=Command.PARENT
    )
    
    
"""
NOTE:
Setting `graph` to `Command.PARENT` navigates to the closest parent graph.
If you update a key shared by both parent and subgraph state schemas, you must define a reducer for that key in the parent graph state.
"""

# Example: Multi-Agent Handoff
# In multi-agent scenarios, you may want to transfer control and state to another agent node.
# Use Command to update the state (e.g., mark a handoff) and route to the target agent.
def agent_handoff(state) -> Command[Literal["agent_b"]]:
    # Indicate a handoff and route to agent_b
    return Command(
        update={"handoff": True},  # Mark that a handoff occurred
        goto="agent_b"             # Route to the agent_b node
    )

# Example: Using Command Inside Tools
# Tools can update the graph state and control flow after performing an operation.
# For example, after looking up customer information, update the state and route to the next node.
def lookup_customer(state) -> Command[Literal["next_node"]]:
    # Simulate fetching customer information (replace with actual lookup logic)
    customer_info = {"name": "Alice", "id": 1234}
    return Command(
        update={"customer": customer_info},  # Store customer info in state
        goto="next_node"                     # Route to the next node for further processing
    )


# Human-in-the-Loop Workflows with Command and Interrupts
def request_approval(state) -> Command[Literal["process_approval"]]:
    """
    Triggers an interrupt for human approval.
    """
    return Command()  # This node would trigger an interrupt externally

# To resume after an interrupt:
resume_values = {"approval_request": "Approved"}  # User feedback

# Resume graph execution with the provided feedback
app.invoke(Command(resume=resume_values), config=config)
# The graph continues from the point of interruption using the feedback.
