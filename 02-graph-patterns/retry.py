import sqlite3
from langchain.chat_models import init_chat_model
from langgraph.graph import END, MessagesState, StateGraph, START
from langgraph.types import RetryPolicy
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

# Initialize the database and model
db = SQLDatabase.from_uri("sqlite:///:memory:")
modal = ChatOllama(model="llama3.2:3b", temperature=0, streaming=True)

def query_database(state: MessagesState):
    """
    Query the database.
    This function may raise sqlite3.OperationalError, which we want to retry on.
    """
    query_result = db.run("SELECT * FROM Artist LIMIT 10;")
    return {"messages": [AIMessage(content=query_result)]}

def call_model(state: MessagesState):
    """
    Call the LLM model.
    This function may fail transiently, so we set a max retry attempts.
    """
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# Build the workflow graph
builder = StateGraph(MessagesState)

# Add nodes with retry policies
builder.add_node(
    "query_database",
    query_database,
    retry_policy=RetryPolicy(retry_on=sqlite3.OperationalError),  # Retry only on OperationalError
)

# RetryPolicy options:
# - max_attempts: Sets how many times to retry (default: 3)
# - retry_on: Specify exception(s) to trigger a retry (default: Exception)
# - backoff_factor: Controls exponential backoff timing (default: 1), which means the wait time between retries increases exponentially.
#  ==> For example, with backoff_factor=2, the delay doubles after each failed attempt.

builder.add_node(
    "model",
    call_model,
    retry_policy=RetryPolicy(max_attempts=5),  # Retry up to 5 times on most exceptions
)

# Define workflow edges
builder.add_edge(START, "model")
builder.add_edge("model", "query_database")
builder.add_edge("query_database", END)

# Compile the graph
graph = builder.compile()
