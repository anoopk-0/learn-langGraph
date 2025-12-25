"""
LangGraph Parallel Pattern 

 The parallel pattern allows multiple nodes to run at the same time in a single super-step.
 This is useful for tasks like batch processing, running multiple agents/tools, or aggregating results.

 How it works:
 - After a node runs, the routing function can send the state to multiple nodes in parallel.
 - All parallel nodes execute, then the graph continues to the next step.
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict

class ParallelState(TypedDict):
    a: int
    b: int
    sum: int

# Node: Set value for 'a'
def set_a(state: ParallelState) -> ParallelState:
    state['a'] = 2
    return state

# Node: Set value for 'b'
def set_b(state: ParallelState) -> ParallelState:
    state['b'] = 3
    return state

# Node: Aggregate results
def aggregate(state: ParallelState) -> ParallelState:
    state['sum'] = state['a'] + state['b']
    return state

# Routing function: after start, run both set_a and set_b in parallel
def run_parallel(state: ParallelState):
    return ["set_a", "set_b"]

workflow = StateGraph(ParallelState)
workflow.add_node("set_a", set_a)
workflow.add_node("set_b", set_b)
workflow.add_node("aggregate", aggregate)
workflow.set_entry_point("aggregate_start")
workflow.add_node("aggregate_start", lambda state: state)  # dummy node to start parallel
workflow.add_conditional_edges(
    "aggregate_start",
    run_parallel,
    {
        "set_a": "set_a",
        "set_b": "set_b"
    }
)
workflow.add_edge("set_a", "aggregate")
workflow.add_edge("set_b", "aggregate")
workflow.add_edge("aggregate", END)

app = workflow.compile()

if __name__ == "__main__":
    state = {"a": 0, "b": 0, "sum": 0}
    result = app.invoke(state)
    print(f"Parallel finished: a={result['a']}, b={result['b']}, sum={result['sum']}")
