from typing import Any, Dict, Optional, Sequence, Annotated
from pymongo import MongoClient
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama

# --- MongoDB-backed Checkpointer Implementation ---
class MongoDBCheckpointSaver(BaseCheckpointSaver):
    """
    Checkpointer for LangGraph that stores checkpoints in a MongoDB collection.
    Enables persistent, scalable state management for graph workflows.
    """
    def __init__(self, uri: str, db_name: str = "langgraph", collection_name: str = "checkpoints"):
        # Connect to MongoDB and select database/collection
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def put(self, config: Dict[str, Any], metadata: Dict[str, Any], values: Dict[str, Any], next_nodes: Any, tasks: Any) -> str:
        """
        Save a checkpoint to MongoDB. Returns the inserted checkpoint's ID.
        """
        checkpoint = {
            "config": config,
            "metadata": metadata,
            "values": values,
            "next": next_nodes,
            "tasks": tasks
        }
        result = self.collection.insert_one(checkpoint)
        return str(result.inserted_id)

    def get_tuple(self, config: Dict[str, Any], checkpoint_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve a checkpoint by thread_id and optional checkpoint_id.
        """
        query = {"config.thread_id": config.get("thread_id")}
        if checkpoint_id:
            query["_id"] = checkpoint_id
        checkpoint = self.collection.find_one(query)
        return checkpoint

    def list(self, config: Dict[str, Any]) -> list:
        """
        List all checkpoints for a given thread_id.
        """
        query = {"config.thread_id": config.get("thread_id")}
        return list(self.collection.find(query))

    def delete_thread(self, thread_id: str) -> int:
        """
        Delete all checkpoints for a given thread_id. Returns count of deleted documents.
        """
        result = self.collection.delete_many({"config.thread_id": thread_id})
        return result.deleted_count

# --- LangGraph Workflow Example Using MongoDBCheckpointSaver ---

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

# Replace with your MongoDB URI
mongodb_uri = "mongodb://localhost:27017/"
checkpointer = MongoDBCheckpointSaver(uri=mongodb_uri)

# Compile the agent with MongoDB-backed checkpointing
agent = workflow_build.compile(checkpointer=checkpointer)

# --- Conversation Loop ---
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    # Pass the user message to the agent and persist state in MongoDB
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config={
        "configurable": {
            "thread_id": "1",
            "user_id": "1"
        }
    })
    # Print the assistant's response
    if response and 'messages' in response and response['messages']:
        print("Assistant:", response['messages'][-1].content)

# --- Checkpoint Retrieval Examples ---
# Get the latest checkpoint for a thread
latest_checkpoint = agent.get_state({"configurable": {"thread_id": "1"}})
print("Latest checkpoint:", latest_checkpoint)

# List all checkpoints for a thread
history = agent.get_state_history({"configurable": {"thread_id": "1"}})
for checkpoint in history:
    print("Checkpoint:", checkpoint)
