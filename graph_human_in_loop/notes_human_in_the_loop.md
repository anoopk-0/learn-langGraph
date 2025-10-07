## Human-in-the-Loop (HITL) Design Patterns in LangGraph

LangGraph enables flexible workflow orchestration for LLM applications. Human-in-the-Loop (HITL) design patterns in LangGraph allow you to insert human decision points into automated pipelines, ensuring quality, safety, and adaptability.

**Typical HITL Workflow in LangGraph**

1. **Automated Node(s):** LLM or tool nodes generate outputs or predictions.
2. **Human Review Node:** A node is dedicated to human validation, correction, or approval of outputs.
3. **Branching Logic:** The workflow can branch based on human feedback (e.g., approve, edit, reject).
4. **Feedback Integration:** Human corrections are fed back into the system for retraining or future improvement.

### Design Patterns

#### 1. Human Review Node

Insert a node in the graph where human input is required. This node can:

- Display model output to a user
- Collect feedback, edits, or approval
- Pass the result to downstream nodes

#### 2. Conditional Branching

Use branching logic to route workflow based on human decisions:

- If approved, continue to next automated step
- If rejected, return to previous node or escalate for further review

#### 3. Feedback Loop

Store human corrections and use them to retrain or fine-tune the model, closing the loop for continuous improvement.

### Example: HITL Pattern in LangGraph

```python
from langgraph.graph import StateGraph

class MyState(dict):
	pass

def llm_node(state):
	# Simulate LLM output
	return {"output": "Draft response"}

def human_review_node(state):
	# Simulate human review (replace with UI or input logic)
	print(f"Model output: {state['output']}")
	decision = input("Approve (a) / Edit (e) / Reject (r): ")
	if decision == "a":
		return {"approved": True, "output": state["output"]}
	elif decision == "e":
		edited = input("Enter your edit: ")
		return {"approved": True, "output": edited}
	else:
		return {"approved": False}

graph = StateGraph(MyState)
graph.add_node("llm", llm_node)
graph.add_node("human_review", human_review_node)
graph.add_edge("llm", "human_review")
graph.set_entry_point("llm")
compiled_graph = graph.compile()

result = compiled_graph.invoke({})
print("Final result:", result)
```

### Best Practices

- Make human review nodes user-friendly and context-rich
- Log human decisions for audit and retraining
- Use branching to handle different review outcomes
- Integrate feedback for model improvement

### References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
