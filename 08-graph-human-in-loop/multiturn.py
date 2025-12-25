from langgraph.graph import StateGraph, END, add_messages
from langgraph.types import Command, interrupt
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import uuid

# Initialize the LLM (Ollama)
llm = ChatOllama(model="llama3.2:3b", temperature=0)

# Define the state structure for the workflow
class State(TypedDict):
    linkedin_topic: str
    generated_post: Annotated[List[str], add_messages]
    human_feedback: Annotated[List[str], add_messages]

# Node: Generate LinkedIn post using LLM and previous feedback
def model(state: State) -> dict:
    """
    Uses the LLM to generate a LinkedIn post, incorporating the latest human feedback.
    Returns updated generated_post and preserves human_feedback.
    """
    print("[model] Generating content...")
    linkedin_topic = state["linkedin_topic"]
    feedback = state["human_feedback"] if "human_feedback" in state else ["No Feedback yet"]
    prompt = f"""
        LinkedIn Topic: {linkedin_topic}
        Human Feedback: {feedback[-1] if feedback else 'No feedback yet'}
        Generate a structured and well-written LinkedIn post based on the given topic.
        Consider previous human feedback to refine the response.
    """
    response = llm.invoke([
        SystemMessage(content="You are an expert LinkedIn content writer"),
        HumanMessage(content=prompt)
    ])
    generated_post = response.content
    print(f"[model_node] Generated post:\n{generated_post}\n")
    return {
        "generated_post": [AIMessage(content=generated_post)],
        "human_feedback": feedback
    }

# Node: Human-in-the-loop for feedback and control flow
def human_node(state: State) -> Command:
    """
    Pauses execution for human feedback using interrupt().
    If user types 'done', transitions to end_node; otherwise, loops back to model with updated feedback.
    """
    print("\n[human_node] Awaiting human feedback...")
    generated_post = state["generated_post"]
    user_feedback = interrupt({
        "generated_post": generated_post,
        "message": "Provide feedback or type 'done' to finish"
    })
    print(f"[human_node] Received human feedback: {user_feedback}")
    if user_feedback.lower() == "done":
        return Command(update={"human_feedback": state["human_feedback"] + ["Finalised"]}, goto="end_node")
    return Command(update={"human_feedback": state["human_feedback"] + [user_feedback]}, goto="model")

# Node: End node to display final results
def end_node(state: State) -> dict:
    """
    Final node: displays the final generated post and all human feedback.
    """
    print("\n[end_node] Process finished")
    print("Final Generated Post:", state["generated_post"][-1])
    print("Final Human Feedback:", state["human_feedback"])
    return {
        "generated_post": state["generated_post"],
        "human_feedback": state["human_feedback"]
    }

# Build the workflow graph
graph = StateGraph(State)
graph.add_node("model", model)
graph.add_node("human_node", human_node)
graph.add_node("end_node", end_node)
graph.set_entry_point("model")
graph.add_edge("model", "human_node")
graph.add_edge("human_node", "model")
graph.set_finish_point("end_node")

# Enable persistence and interrupt mechanism
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# Get initial topic from user
linkedin_topic = input("Enter your LinkedIn topic: ")
initial_state = {
    "linkedin_topic": linkedin_topic,
    "generated_post": [],
    "human_feedback": []
}

# Main loop: stream events and handle interrupts for human feedback
for chunk in app.stream(initial_state, config=thread_config):
    for node_id, value in chunk.items():
        if node_id == "__interrupt__":
            while True:
                user_feedback = input("Provide feedback (or type 'done' when finished): ")
                # Resume graph execution with user's feedback
                app.invoke(Command(resume=user_feedback), config=thread_config)
                if user_feedback.lower() == "done":
                    break


