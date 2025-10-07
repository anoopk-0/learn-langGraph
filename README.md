## What is an LLM?

LLM stands for Large Language Model. These are advanced AI models trained on vast amounts of text data to understand, generate, and interact using human language. Examples include OpenAI's GPT series, Meta's Llama, and Google's Gemini.

You can run LLMs via cloud APIs (like OpenAI, Anthropic, Google) or locally using open-source models. Running locally requires sufficient hardware (RAM, GPU/CPU) and frameworks like Ollama, LM Studio, or Hugging Face Transformers.

#### Best LLMs for Local Use & Data Preservation

- **Llama 3 (Meta):** Available in 8B, 70B sizes. Good performance, open weights, strong community support.
- **Mistral:** Efficient, fast, available in 7B, 8x22B sizes. Good for resource-constrained environments.
- **Phi-3 (Microsoft):** Small, efficient, good for edge devices.
- **Gemma (Google):** Lightweight, privacy-focused.

For data preservation, prefer models you run locally (no cloud API), such as Llama 3, Mistral, or Phi-3. Choose model size based on your hardware: 7B-8B for laptops, 13B+ for desktops/servers with more RAM.

#### What is Hugging Face?

Hugging Face is a leading platform and community for open-source AI models, especially LLMs. It provides the Transformers library, which allows easy access to thousands of pre-trained models for natural language processing, computer vision, and more. Hugging Face hosts model repositories, datasets, and tools for training, deploying, and sharing models. It is widely used for running LLMs locally, fine-tuning models, and integrating AI into applications.

Key features:

- Access to popular LLMs (Llama, Mistral, GPT, etc.)
- Transformers library for easy model use and deployment
- Model Hub for sharing and discovering models
- Community support and documentation

## What is LangGraph?

LangGraph is a Python framework for building stateful, multi-agent, and graph-based workflows on top of LLMs. It enables complex reasoning, memory, and tool use by connecting nodes (agents/functions) in a directed graph.

### What is LangChain?

LangChain is a popular framework for developing LLM-powered applications. It provides abstractions for chains, agents, memory, and integrations with various LLMs, tools, and data sources.

#### Advantages Over Other Frameworks

- **LangGraph:**

  - Graph-based workflows allow flexible, non-linear reasoning.
  - Supports multi-agent collaboration and persistent memory.
  - Integrates with LangChain and other LLM tools.

- **LangChain:**
  - Rich ecosystem, many integrations (tools, databases, APIs).
  - Modular design for chaining prompts, agents, and memory.
  - Large community and extensive documentation.

Compared to other frameworks, LangGraph and LangChain offer more flexibility, modularity, and support for advanced LLM use cases (tool use, memory, multi-agent systems).

## LLM Parameters: Temperature and More

When using LLMs, you can fine-tune their responses by adjusting several important parameters. Here’s a detailed explanation of each, along with practical examples:

- **Temperature:** Determines the randomness of the model’s output. Lower values (e.g., 0.1–0.3) make responses more focused and predictable, while higher values (e.g., 0.7–1.0) encourage creativity and diversity.

  - _Example:_
    - `temperature=0.2` (very deterministic, suitable for factual answers)
    - `temperature=0.9` (more creative, good for brainstorming or storytelling)

- **Top-p (nucleus sampling):** Limits the model to selecting from the smallest set of tokens whose cumulative probability is at least top-p. This helps balance diversity and coherence.

  - _Example:_
    - `top_p=0.8` (only considers tokens that together make up 80% of the probability mass)

- **Top-k:** Restricts the model to choosing from the top k most likely tokens at each step. Useful for controlling randomness and output diversity.

  - _Example:_
    - `top_k=50` (model chooses from the 50 most probable next tokens)

- **Max tokens:** Sets the maximum number of tokens the model can generate in its response. This controls the length and cost of the output.

  - _Example:_
    - `max_tokens=100` (response will not exceed 100 tokens)

- **Repetition penalty:** Penalizes repeated phrases or words, making the output less repetitive.

  - _Example:_
    - `repetition_penalty=1.2` (higher values discourage repetition)

- **Presence penalty / Frequency penalty:** These parameters further control repetition and encourage the model to introduce new topics or avoid repeating the same words.
  - _Example:_
    - `presence_penalty=0.5` (encourages new topics)
    - `frequency_penalty=0.3` (reduces repeated words)

**Practical Example (OpenAI API):**

```python
response = openai.ChatCompletion.create(
		model="gpt-4",
		messages=[{"role": "user", "content": "Write a short poem about the ocean."}],
		temperature=0.8,
		max_tokens=60,
		top_p=0.95,
		frequency_penalty=0.4,
		presence_penalty=0.6
)
```

**Practical Example (Hugging Face Transformers):**

```python
output = model.generate(
		input_ids,
		max_length=60,
		temperature=0.8,
		top_p=0.95,
		top_k=40,
		repetition_penalty=1.2
)
```

By experimenting with these parameters, you can tailor the LLM’s output to suit your specific needs—whether you want concise, factual answers or creative, varied responses.
