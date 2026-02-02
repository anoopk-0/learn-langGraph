"""
Retry:
Retry means automatically running a failed operation again instead of stopping the program.

Why Retry is Needed:
Sometimes a step fails because of temporary issues (network glitch, timeout, API error).
Retry allows the system to try again without crashing.

RetryPolicy:
RetryPolicy defines the rules for retrying a failed node.

Main RetryPolicy Options:
- max_attempts → How many times to try again
- retry_on → Which exceptions should trigger retry
- backoff_factor → Delay growth between retries

Example:
RetryPolicy(max_attempts=3, retry_on=(Exception,))

Meaning:
Try up to 3 times when any exception occurs.

How Retry Works in LangGraph:
- Node runs
- If exception occurs → retry
- If still fails → retry again (until max_attempts)
- If still failing → graph stops with error

Important Rule:
Retry happens only when a node raises an exception.

Where RetryPolicy is Applied:
RetryPolicy is attached to a node when adding it to the graph.

graph.add_node("node_name", node_function, retry=retry_policy)

One-Line Summary:
RetryPolicy makes LangGraph nodes fault-tolerant by automatically re-running them when errors happen.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

# ---- State ----
class AgentState(TypedDict):
    number: int


# ---- Global failure flag ----
fail_once = True


# ---- Nodes ----
def print_node(state: AgentState):
    print(f"number is {state['number']}")
    return {}


def decrement_node(state: AgentState):
    global fail_once

    if fail_once:
        fail_once = False
        print("❌ Simulated failure")
        raise ValueError("temporary failure")

    print("✅ decrement_node succeeded")
    return {"number": state["number"] - 1}


def check_node(state: AgentState):
    if state["number"] == 0:
        return "stop"
    return "loop"


# ---- Retry Policy (IMPORTANT FIX) ----
retry_policy = RetryPolicy(
    max_attempts=3,
    retry_on=(Exception,)
)


# ---- Graph ----
graph = StateGraph(AgentState)

graph.add_node("print_node", print_node, retry=retry_policy)
graph.add_node("decrement_node", decrement_node, retry=retry_policy)

graph.add_edge(START, "print_node")

graph.add_conditional_edges(
    "print_node",
    check_node,
    {
        "loop": "decrement_node",
        "stop": END
    }
)

graph.add_edge("decrement_node", "print_node")

app = graph.compile()


# ---- Run ----
result = app.invoke({"number": 5})
print("Final:", result)
