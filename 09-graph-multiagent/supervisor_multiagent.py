import os
from typing import Literal
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
from langchain_experimental.tools import PythonREPLTool

from langgraph.types import Command
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent

# Set API keys and model configuration
os.environ["TAVILY_API_KEY"] = "your_tavily_api_key_here"
llm = ChatOllama(model="llama3.2:3b", temperature=0)
tavily_search = TavilySearch(max_results=2)
python_repl_tool = PythonREPLTool()

# Supervisor schema
class Supervisor(BaseModel):
    next: Literal["enhancer", "researcher", "coder"] = Field(
        description="Which specialist to activate next: 'enhancer', 'researcher', or 'coder'."
    )
    reason: str = Field(
        description="Justification for the routing decision."
    )

# ------------------- NODE DEFINITIONS FOR MULTI-AGENT GRAPH -------------------

def supervisor_node(state: MessagesState) -> Command[Literal["enhancer", "researcher", "coder"]]:
    """
    Supervisor node: Decides which specialist agent to activate next.
    Uses a system prompt to instruct the LLM to choose between Enhancer, Researcher, or Coder,
    and provide a rationale for the choice.
    """
    system_prompt = (
        "You are a workflow supervisor managing three agents: Enhancer, Researcher, and Coder. "
        "Choose the next agent based on the current state.\n"
        "Enhancer: Clarifies and improves queries.\n"
        "Researcher: Gathers facts and context.\n"
        "Coder: Implements solutions and performs computations.\n"
        "Always provide a clear rationale for your choice."
    )
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Supervisor).invoke(messages)
    goto = response.next
    reason = response.reason
    return Command(
        update={
            "messages": [HumanMessage(content=reason, name="supervisor")]
        },
        goto=goto,
    )

def enhancer_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    """
    Enhancer node: Refines and clarifies the user's query.
    Uses a system prompt to instruct the LLM to improve the query without asking questions back.
    """
    system_prompt = (
        "You are a Query Refinement Specialist. Improve and clarify the user's query. "
        "Do not ask questions back; make reasonable assumptions and produce a precise, actionable request."
    )
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    enhanced_query = llm.invoke(messages)
    return Command(
        update={
            "messages": [HumanMessage(content=enhanced_query.content, name="enhancer")]
        },
        goto="supervisor",
    )

def research_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Researcher node: Gathers relevant information and context.
    Uses a system prompt and the TavilySearch tool to collect and organize findings.
    """
    research_prompt = (
        "You are an Information Specialist. Gather relevant, accurate, and up-to-date information. "
        "Organize findings clearly and cite sources when possible."
    )
    research_agent = create_react_agent(
        llm,
        tools=[tavily_search]
    )
    # Prepend system prompt to messages
    state_with_prompt = state.copy()
    state_with_prompt["messages"] = [
        {"role": "system", "content": research_prompt}
    ] + state["messages"]
    result = research_agent.invoke(state_with_prompt)
    return Command(
        update={
            "messages": [HumanMessage(content=result["messages"][-1].content, name="researcher")]
        },
        goto="validator",
    )

def code_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Coder node: Performs calculations, code execution, and technical problem-solving.
    Uses a system prompt and the PythonREPLTool for code-related tasks.
    """
    code_prompt = (
        "You are a coder and analyst. Focus on calculations, code execution, and technical problem-solving."
    )
    code_agent = create_react_agent(
        llm,
        tools=[python_repl_tool]
    )
    # Prepend system prompt to messages
    state_with_prompt = state.copy()
    state_with_prompt["messages"] = [
        {"role": "system", "content": code_prompt}
    ] + state["messages"]
    result = code_agent.invoke(state_with_prompt)
    return Command(
        update={
            "messages": [HumanMessage(content=result["messages"][-1].content, name="coder")]
        },
        goto="validator",
    )

# Validator node schema and prompt
validator_prompt = (
    "You are a validator. Review the user's question and the agent's answer. "
    "If the answer is good enough, signal to end the workflow with 'FINISH'. "
    "Only route back to supervisor if the answer is completely off-topic or harmful."
)

class Validator(BaseModel):
    next: Literal["supervisor", "FINISH"] = Field(
        description="Next step: 'supervisor' to continue, 'FINISH' to end."
    )
    reason: str = Field(
        description="Reason for the decision."
    )

def validator_node(state: MessagesState) -> Command[Literal["supervisor", "__end__"]]:
    """
    Validator node: Reviews the agent's answer and decides whether to end the workflow or route back to supervisor.
    Uses a system prompt and structured output for decision making.
    """
    user_question = state["messages"][0].content
    agent_answer = state["messages"][-1].content
    messages = [
        {"role": "system", "content": validator_prompt},
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": agent_answer},
    ]
    response = llm.with_structured_output(Validator).invoke(messages)
    goto = response.next
    reason = response.reason
    if goto == "FINISH":
        goto = END
    return Command(
        update={
            "messages": [HumanMessage(content=reason, name="validator")]
        },
        goto=goto,
    )
    
    
graph = StateGraph(MessagesState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("enhancer", enhancer_node)
graph.add_node("researcher", research_node)
graph.add_node("coder", code_node)
graph.add_node("validator", validator_node)

graph.add_edge(START, "supervisor")

app = graph.compile()

inputs = {
    "messages": [
        ("user", "Weather in Chennai"),
    ]
}

# Stream workflow execution and print updates from each node
for event in app.stream(inputs):
    for node, state in event.items():
        if state is None:
            continue
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            print(f"Update from '{node}': {last_message.name}: {last_message.content}\n")
