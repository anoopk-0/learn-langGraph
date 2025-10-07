from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt
from typing import TypedDict
from langgraph.checkpoint.memory import MemorySaver

# Initialize memory checkpointer for graph persistence
memory = MemorySaver()

# Define the state structure for the workflow
class State(TypedDict):
    value: str

# Node A: Appends 'a' to state value and routes to node B
def node_a(state: State) -> Command:
    print("Node A: Processing...")
    return Command(
        goto="node_b",
        update={"value": state["value"] + "a"}
    )

# Node B: Interrupts for human decision, then routes to C or D
def node_b(state: State) -> Command:
    print("Node B: Awaiting human input...")
    human_response = interrupt("Do you want to go to C or D? Type C/D")
    print("Human Review Value:", human_response)
    next_node = "node_c" if human_response == "C" else "node_d"
    return Command(
        goto=next_node,
        update={"value": state["value"] + "b"}
    )

# Node C: Appends 'c' to state value and ends workflow
def node_c(state: State) -> Command:
    print("Node C: Finalizing...")
    return Command(
        goto=END,
        update={"value": state["value"] + "c"}
    )

# Node D: Appends 'd' to state value and ends workflow
def node_d(state: State) -> Command:
    print("Node D: Finalizing...")
    return Command(
        goto=END,
        update={"value": state["value"] + "d"}
    )

# Build the workflow graph
graph = StateGraph(State)
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)
graph.add_node("node_d", node_d)
graph.set_entry_point("node_a")

# Compile the app with memory checkpointer
graph_app = graph.compile(checkpointer=memory)

# Configuration for thread persistence
config = {"configurable": {"thread_id": "1"}}

# Initial state for the workflow
initial_state = {"value": ""}

# First run: triggers interrupt at node B
first_result = graph_app.invoke(initial_state, config, stream_mode="updates")
print("First Run Result:", first_result)

# Resume after human input (simulate choosing 'C')
second_result = graph_app.invoke(Command(resume="C"), config=config, stream_mode="updates")
print("Second Run Result:", second_result)