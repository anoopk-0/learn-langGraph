"""
Prompt engineering is the discipline of designing, refining, and structuring prompts to maximize the quality, relevance, and reliability of outputs from LLMs. Effective prompt engineering is essential for tasks such as conversation, summarization, coding, reasoning, and more.
"""
# 1. Message Types in Chat-Based AI
# ---------------------------------
# System Message: Sets rules, persona, and behavior of the AI.
# Human Message: The user's query or instruction.
# Assistant Message: The model's response, shaped by previous messages.

def build_prompt(role: str, instruction: str, user_query: str) -> str:
    """
    Build a structured prompt for a chat-based LLM.
    Args:
        role: The persona or expertise for the AI (system message).
        instruction: Instructions or constraints for the AI.
        user_query: The user's question or task.
    Returns:
        A formatted prompt string.
    """
    system_msg = f"System: {role}. {instruction}"
    human_msg = f"Human: {user_query}"
    return f"{system_msg}\n{human_msg}"

# Example: System, Human, and Assistant messages
example_prompt = build_prompt(
    role="You are a knowledgeable, friendly assistant",
    instruction="Always provide clear, step-by-step answers and avoid making up information.",
    user_query="List three key differences between supervised and unsupervised learning in a table."
)
print("\n--- Example: System & Human Message ---\n", example_prompt)

# Simulated Assistant Response (for illustration)
assistant_response = '''
| Feature           | Supervised Learning   | Unsupervised Learning |
|-------------------|----------------------|----------------------|
| Data              | Labeled              | Unlabeled            |
| Goal              | Predict outcomes     | Find patterns/groups |
| Common Algorithms | SVM, Decision Trees  | Clustering, PCA      |
'''
print("\n--- Example: Assistant Message ---\n", assistant_response)

# 2. Structuring Prompts for Better Responses
# -------------------------------------------
# a. Use clear roles and context
# b. Be specific and explicit
# c. Provide examples (few-shot prompting)
# d. Break down complex tasks (prompt chaining)
# e. Request step-by-step reasoning (chain-of-thought)
# f. Refine iteratively

# Few-shot Prompting Example
few_shot_prompt = """
Q: What is the capital of France?
A: Paris
Q: What is the capital of Japan?
A: Tokyo
Q: What is the capital of Australia?
A:
"""
print("\n--- Few-shot Prompting Example ---\n", few_shot_prompt)

# Prompt Chaining Example (Workflow)
def summarize_document(doc: str) -> str:
    """Simulate summarizing a document."""
    return "Summary: " + doc[:50] + "..."

def extract_key_points(summary: str) -> list:
    """Simulate extracting key points from a summary."""
    return ["Key Point 1", "Key Point 2", "Key Point 3"]

def generate_questions(points: list) -> list:
    """Simulate generating questions from key points."""
    return [f"What about {pt}?" for pt in points]

# Example workflow
doc = "Prompt engineering helps improve LLM outputs by structuring instructions, providing examples, and refining queries."
summary = summarize_document(doc)
key_points = extract_key_points(summary)
questions = generate_questions(key_points)
print("\n--- Prompt Chaining Example ---\nSummary:", summary)
print("Key Points:", key_points)
print("Generated Questions:", questions)

# Chain-of-Thought Prompting Example
cot_prompt = build_prompt(
    role="Show your reasoning step by step.",
    instruction="",
    user_query="If a train travels 60 miles in 1 hour, how far will it travel in 3 hours?"
)
cot_response = "Step 1: The train travels 60 miles in 1 hour. Step 2: In 3 hours, it will travel 60 x 3 = 180 miles."
print("\n--- Chain-of-Thought Prompting ---\nPrompt:\n", cot_prompt)
print("Response:\n", cot_response)

# Iterative Refinement Example
initial_prompt = "Write a function to check if a number is prime."
refined_prompt = """Write a Python function named is_prime. The function should:
- Take one integer argument.
- Return True if the number is prime, False otherwise.
- Handle edge cases (e.g., negative numbers, 0, 1).
- Include a docstring and comments."""
print("\n--- Iterative Refinement Example ---\nInitial Prompt:\n", initial_prompt)
print("Refined Prompt:\n", refined_prompt)

# 3. Advanced Prompt Engineering Strategies
# ----------------------------------------
# a. Prompt Templates
def prompt_template(sentence: str) -> str:
    return f"System: You are a translator.\nHuman: Translate the following sentence to Spanish: {sentence}"
print("\n--- Prompt Template Example ---\n", prompt_template("Good morning!"))

# b. Role-Based Prompting
role_based_prompt = build_prompt(
    role="You are an expert Python developer.",
    instruction="",
    user_query="Write a function to reverse a string."
)
print("\n--- Role-Based Prompting Example ---\n", role_based_prompt)

# c. Self-Consistency Prompting
# (Run multiple prompts and aggregate results; not shown in code)

# d. Multimodal Prompting
# (Combine text with images/audio/code; not shown in code)

# e. Zero-Shot Prompting
zero_shot_prompt = "Human: Translate 'Good morning' to French."
print("\n--- Zero-Shot Prompting Example ---\n", zero_shot_prompt)

# Common Prompting Pitfalls & How to Avoid Them
# ---------------------------------------------
# - Vague Prompts: "Tell me about science." → Too broad. Specify topic and format.
# - Missing Context: "Summarize." → Summarize what? Always provide the source or subject.
# - Overloaded Requests: "Write a story, a poem, and a song about cats." → Split into separate prompts.
# - Unclear Formatting: "Make a table." → Specify columns, rows, and data type.
# - Ambiguous Instructions: "Explain quickly." → Specify word or sentence limits.
