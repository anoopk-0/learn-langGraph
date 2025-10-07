from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages, StateGraph, END
from langchain_ollama import ChatOllama

# Define the state structure for the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the LLM (Ollama with llama3.2:3b model)
llm = ChatOllama(model="llama3.2:3b")

# Node names for the workflow
GENERATE_POST = "generate_post"
GET_REVIEW_DECISION = "get_review_decision"
POST = "post"
COLLECT_FEEDBACK = "collect_feedback"

def generate_post(state: State) -> dict:
    """
    Generate a LinkedIn post using the LLM based on current messages.
    """
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def get_review_decision(state: State) -> str:
    """
    Display the generated post and ask the user for approval.
    Returns the next node name based on user input.
    """
    post_content = state["messages"][-1].content
    print("\n📢 Current LinkedIn Post:\n")
    print(post_content)
    print("\n")
    decision = input("Post to LinkedIn? (yes/no): ")
    return POST if decision.strip().lower() == "yes" else COLLECT_FEEDBACK

def post(state: State) -> None:
    """
    Finalize and 'post' the LinkedIn content.
    """
    final_post = state["messages"][-1].content
    print("\n📢 Final LinkedIn Post:\n")
    print(final_post)
    print("\n✅ Post has been approved and is now live on LinkedIn!")

def collect_feedback(state: State) -> dict:
    """
    Collect feedback from the user to improve the post.
    """
    feedback = input("How can I improve this post? ")
    return {"messages": [HumanMessage(content=feedback)]}

# Build the workflow graph
graph = StateGraph(State)
graph.add_node(GENERATE_POST, generate_post)
graph.add_node(GET_REVIEW_DECISION, get_review_decision)
graph.add_node(COLLECT_FEEDBACK, collect_feedback)
graph.add_node(POST, post)

# Set entry point and transitions
graph.set_entry_point(GENERATE_POST)
graph.add_conditional_edges(GENERATE_POST, get_review_decision)
graph.add_edge(POST, END)
graph.add_edge(COLLECT_FEEDBACK, GENERATE_POST)

# Compile and run the app
app = graph.compile()

if __name__ == "__main__":
    # Initial message to start the workflow
    response = app.invoke({
        "messages": [HumanMessage(content="Write me a LinkedIn post on AI Agents taking over content creation")]
    })
    print(response)
