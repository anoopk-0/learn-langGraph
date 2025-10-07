"""
# InMemorySaver stores all agent state in RAM, so memory is only available
# during the current process/session. No database or file storage is used.
# Suitable for prototyping and temporary sessions.
"""

from langchain_ollama import ChatOllama
from typing import TypedDict, Sequence, Annotated
from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Initialize the LLM (Ollama)
llm = ChatOllama(model="llama3.2:3b", temperature=0)

def llm_chat_node(state: AgentState) -> AgentState:
    """
    Adds a system prompt, invokes the LLM, and appends the response to the state.
    """
    system_prompt = SystemMessage(
        content="You are a helpful assistant. Help the user with their questions and summarize their requests."
    )
    response = llm.invoke([system_prompt] + state["messages"])
    state["messages"].append(response)
    return state

# Build the workflow graph
workflow_build = StateGraph(AgentState)
workflow_build.add_node("llm_chat_node", llm_chat_node)
workflow_build.set_entry_point("llm_chat_node")
workflow_build.set_finish_point("llm_chat_node")

# Use InMemorySaver for short-term, in-process checkpointing
agent = workflow_build.compile(checkpointer=InMemorySaver())

# Get state
"""
# get the latest state snapshot
config = {"configurable": {"thread_id": "1"}}
graph.get_state(config)

# get a state snapshot for a specific checkpoint_id
config = {"configurable": {"thread_id": "1", "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c"}}
graph.get_state(config)
"""

# get history of states
"""
# full history of the graph execution for a given thread id

config = {"configurable": {"thread_id": "1"}}
list(graph.get_state_history(config))
"""

# Replay
"""
config = {"configurable": {"thread_id": "1", "checkpoint_id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}}
graph.invoke(None, config=config)
"""

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    # Pass the user message to the agent
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config={
        # The 'configurable' key allows runtime configuration, e.g. thread_id for multi-session
        "configurable": {
            "thread_id": "1",
            "user_id": "1"
        }
    })
    # Print the assistant's response
    if response and 'messages' in response and response['messages']:
        print("Assistant:", response['messages'][-1].content)