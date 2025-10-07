
# LangGraph Sequential Pattern 
#
# This pattern runs nodes in a strict sequence, where each step depends on the previous one.
#
# Graph Flow:
#   [START] -> [Initialize] -> [Increment] -> [Double] -> [END]
#
# Use Case: Data processing pipelines, multi-step agent workflows, etc.

from langgraph.graph import StateGraph, END
from typing import TypedDict

# State definition: holds the value being processed through the steps
class SeqState(TypedDict):
    value: int

# Step 1: Initialize value
def initialize_node(state: SeqState) -> SeqState:
    """
    Sets the initial value in the state.
    """
    return {"value": 1}

# Step 2: Increment value
def increment_node(state: SeqState) -> SeqState:
    """
    Increments the value by 1.
    """
    return {"value": state["value"] + 1}

# Step 3: Double value
def double_node(state: SeqState) -> SeqState:
    """
    Doubles the value.
    """
    return {"value": state["value"] * 2}

# Build the sequential graph workflow
workflow = StateGraph(SeqState)
workflow.add_node("initialize", initialize_node)   # Step 1
workflow.add_node("increment", increment_node)     # Step 2
workflow.add_node("double", double_node)           # Step 3
workflow.set_entry_point("initialize")
workflow.add_edge("initialize", "increment")
workflow.add_edge("increment", "double")
workflow.add_edge("double", END)

# Compile and run the graph
app = workflow.compile()

if __name__ == "__main__":
    print("--- Sequential Pattern Example ---")
    result = app.invoke({})
    print("Final state:", result)  # Should print: {'value': 4}
