"""
# LangGraph Conditional Pattern

# The conditional pattern allows a graph to dynamically choose which node to execute next based on the current state.
#
# How it works:
# - After a node runs, a routing function examines the state and decides which node(s) should run next.
# - This enables branching logic, so different paths can be taken depending on user input or workflow results.
# - Common use cases: decision trees, multi-step forms, agent workflows with branching, tool selection, etc.
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# State definition: holds numbers, operation, and result
class AgentState(TypedDict):
    number_1: int
    number_2: int
    operation: str
    result: int
    
# Node: Addition
def add_node(state: AgentState) -> AgentState:
    state['result'] = state['number_1'] + state['number_2']
    return state

# Node: Subtraction
def subtract_node(state: AgentState) -> AgentState:
    state['result'] = state['number_1'] - state['number_2']
    return state

# Node: Multiplication
def multiply_node(state: AgentState) -> AgentState:
    state['result'] = state['number_1'] * state['number_2']
    return state

# Routing function: decides which node to run next based on 'operation'
def route_operation(state: AgentState):
    op = state['operation']
    if op == '+':
        return "add"
    elif op == '-':
        return "subtract"
    elif op == '*':
        return "multiply"
    else:
        raise ValueError(f"Unknown operation: {op}")


# Build the conditional graph
workflow = StateGraph(AgentState)
workflow.add_node("add", add_node)
workflow.add_node("subtract", subtract_node)
workflow.add_node("multiply", multiply_node)
workflow.add_node("router", lambda state: state)  # Pass-through node for routing

workflow.add_edge(START, "router")

# Conditional edges: route to the correct operation node
workflow.add_conditional_edges(
    "router",
    route_operation,
    {
        "add": "add",
        "subtract": "subtract",
        "multiply": "multiply"
    }
)
workflow.add_edge("add", END)
workflow.add_edge("subtract", END)
workflow.add_edge("multiply", END)

# Compile the graph
app = workflow.compile()


# Example usage: run the graph with user input
if __name__ == "__main__":
    state = {}
    state['operation'] = input("Enter operation (+, -, *): ")
    state['number_1'] = int(input("Enter first number: "))
    state['number_2'] = int(input("Enter second number: "))

    result = app.invoke(AgentState(**state))
    print(f"Result of {state['number_1']} {state['operation']} {state['number_2']} = {result['result']}")

    # Print graph as Mermaid text (copy to https://mermaid.live/ for visualization)
    print(workflow.get_graph().draw_mermaid())

