# Multi-Agent Systems in LangGraph

Multi-agent systems use multiple independent agents (often powered by LLMs) to manage complex workflows, modularize logic, and enable specialization. LangGraph provides flexible architectures for building, scaling, and controlling multi-agent systems.

**Why Multi-Agent?**
- **Modularity:** Easier development, testing, and maintenance.
- **Specialization:** Expert agents for specific domains.
- **Control:** Explicit communication and routing between agents.

### Architectures

Below are typical architectures for multi-agent systems in LangGraph, illustrated with diagrams:

#### 1. Network

Agents (nodes) can communicate freely with any other agent, enabling flexible, non-hierarchical workflows.

```
[Agent A]───┬───[Agent B]
    │      │
    └──────┴───[Agent C]
```

#### 2. Supervisor

A central supervisor agent (LLM) routes tasks to specialized agents, supporting parallelism and map-reduce patterns.

```
        [Supervisor]
         /   |   \
 [Agent X] [Agent Y] [Agent Z]
```

#### 3. Supervisor (Tool-Calling)

Supervisor agent calls sub-agents exposed as tools, using a tool-calling LLM loop.

```
[Supervisor]───▶[Tool-Agent 1]
     │
     └──────▶[Tool-Agent 2]
```

#### 4. Hierarchical

Teams of agents are managed by individual supervisors, with a top-level supervisor coordinating the teams.

```
        [Top Supervisor]
         /         \
 [Supervisor A]   [Supervisor B]
    /   \            |
[Agent 1][Agent 2] [Agent 3]
```

#### 5. Custom Workflow

Agents are connected in explicit or dynamic sequences, allowing for tailored control flows.

```
[Agent 1]──▶[Agent 2]──▶[Agent 3]
        ▲           │
        └───────────┘
```

### Handoffs & Agent-to-Agent (A2A) Communication

**Handoff** refers to the process where one agent passes control and relevant state (data, context, or payload) to another agent. This enables agents to collaborate, delegate tasks, or specialize in different parts of a workflow. In LangGraph, handoffs are explicit: an agent decides which agent should handle the next step and what information to pass along.

**Agent-to-Agent (A2A) communication** is a direct interaction between agents, where one agent calls another, often using a handoff. This can be peer-to-peer (network architecture) or mediated by a supervisor (supervisor or hierarchical architectures).

#### Example: Basic Handoff

Suppose Agent A completes its part of a task and wants Agent B to continue. Agent A uses the `Command` object to specify the next agent and any state updates:

```python
from langgraph.graph import Command
from typing import Literal

def agent_a(state) -> Command[Literal["agent_b"]]:
    # Decide to hand off to agent_b
    next_agent = "agent_b"
    # Pass updated state
    return Command(goto=next_agent, update={"progress": "step_completed_by_a"})
```

Agent B receives the updated state and continues processing.

#### Example: A2A Communication in a Network

In a network architecture, agents can communicate freely. For instance, Agent X may decide to call Agent Y based on the current state:

```python
def agent_x(state) -> Command[Literal["agent_y", "agent_z"]]:
    if state["task"] == "analyze":
        return Command(goto="agent_y", update={"analysis": "data"})
    else:
        return Command(goto="agent_z", update={"review": "data"})
```

#### Handoffs in Subgraphs

When agents are organized in subgraphs (nested workflows), you can route handoffs between agents in different graphs using `graph=Command.PARENT`:

```python
def agent_sub(state) -> Command[Literal["agent_main"]]:
    # Handoff to an agent in the parent graph
    return Command(goto="agent_main", update={"result": "subgraph_done"}, graph=Command.PARENT)
```

#### Tool-Calling as Handoff

Agents can be exposed as tools, allowing supervisors or other agents to call them and pass state:

```python
from langchain_core.tools import tool

@tool
def handoff_to_bob():
    # Handoff to agent 'bob' in the parent graph
    return Command(goto="bob", update={"info": "from_tool"}, graph=Command.PARENT)
```

**Summary:**  
- Handoff = passing control and state from one agent to another.
- A2A = direct agent-to-agent communication, often via handoff.
- Use `Command` to specify the next agent and state updates.
- Handoffs can occur within a graph or across subgraphs.
- Tool-calling is a special case of handoff, wrapping agents as callable tools.

## Example Architectures

- **Network:** Agents choose which agent to call next or when to finish, enabling flexible workflows.
- **Supervisor:** A central supervisor delegates tasks to agents, who return results to the supervisor.
- **Tool-Calling:** The supervisor LLM invokes agent tools in a loop, passing control and state as needed.
- **Hierarchical:** Multiple supervisors manage teams of agents, coordinated by a top-level supervisor.
- **Custom Workflow:** Explicit or dynamic routing between agents using `Command` objects.

## Communication & State Management

Agents communicate by sharing a message list (e.g., `messages`). Key considerations include:

- **Handoffs vs Tool Calls:** Agents transfer state via handoffs or tool call arguments.
- **Message History:** Share either the full message history (scratchpad) or just final results. Full history improves reasoning but may require memory management.
- **Agent Identification:** Tag each message with the agent's name (using the `name` parameter or manual tagging).
- **Representing Handoffs:** Add tool messages to mark successful handoffs between agents.
- **Subagent State:** Maintain private scratchpads for each agent, or filter shared messages before passing to subagents.
- **State Schemas:** Use custom schemas for specialized agents, and apply input/output transforms for subgraphs.

LangGraph provides explicit control over agent communication, routing, and state management, making it easy to build scalable multi-agent systems.

