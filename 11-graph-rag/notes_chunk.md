## Chunks

In Natural Language Processing (NLP) and vector databases, a chunk is a small, self-contained segment of a larger text or dataset. Chunks are created to:

- Enable independent processing, such as embedding into vectors
- Facilitate efficient indexing and retrieval
- Preserve semantic coherence, depending on the chosen chunking strategy

Common types of chunks include:

- Sentences
- Paragraphs
- Fixed-length blocks of tokens or characters
- Contextually meaningful units, such as topic-based segments

```txt
> "Artificial Intelligence is transforming industries. Natural Language Processing enables machines to understand human language. Vector databases help store and search embeddings efficiently."

A simple sentence-based chunking would produce:

1. "Artificial Intelligence is transforming industries."
2. "Natural Language Processing enables machines to understand human language."
3. "Vector databases help store and search embeddings efficiently."

Each chunk is a self-contained unit, ready for embedding or retrieval.
```

## Chunking

Chunking refers to splitting large texts into smaller, self-contained segments. This step is crucial in NLP pipelines, especially when working with models that have input size restrictions or when building semantic search and retrieval systems.

#### ✅ Why Chunking Matters:

1. **Token Limit Compliance**  
   Language models (e.g., GPT-4, BERT) have strict token limits. Chunking ensures inputs fit within these boundaries, preventing truncation and errors.

2. **Enhanced Semantic Search**  
   Breaking text into chunks enables more precise and context-aware retrieval in vector databases. Each chunk can be embedded and searched independently for improved relevance.

3. **Efficient Processing & Storage**  
   Smaller chunks are easier to embed, store, and index, resulting in faster processing and reduced resource usage.

4. **Context Preservation**  
   Advanced chunking strategies, such as semantic and recursive chunking, help maintain the integrity of ideas by avoiding splits within sentences or concepts.

### Chunking Strategies

Chunking prepares text for vector databases and Retrieval-Augmented Generation (RAG) pipelines. Below are common strategies with Python code samples:

#### 1. Token-Based Chunking

Splits text into fixed-size token groups using a tokenizer.

```python
from transformers import AutoTokenizer

def token_chunk(text, chunk_size=256):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.tokenize(text)
    chunks = [
        tokenizer.convert_tokens_to_string(tokens[i:i+chunk_size])
        for i in range(0, len(tokens), chunk_size)
    ]
    return chunks

text = "AI is awesome! " * 100
print(token_chunk(text, chunk_size=10))
```

#### 2. Character-Based Chunking

Splits text into chunks of a fixed number of characters.

```python
def char_chunk(text, chunk_size=100):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

text = "AI is awesome! " * 20
print(char_chunk(text, chunk_size=20))
```

#### 3. Sentence-Based Chunking

Splits text by sentences for semantic coherence.

```python
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def sentence_chunk(text):
    return sent_tokenize(text)

text = "AI is awesome! It is changing the world. NLP is a key technology."
print(sentence_chunk(text))
```

#### 4. Paragraph-Based Chunking

Splits text by paragraphs (e.g., double newlines).

```python
def paragraph_chunk(text):
    return [p for p in text.split('\n\n') if p.strip()]

text = "AI is awesome!\n\nNLP is a key technology.\n\nVector databases are useful."
print(paragraph_chunk(text))
```

#### 5. Semantic/Recursive Chunking

Splits text at logical boundaries (paragraphs, sentences, tokens) to maximize coherence and fit within limits.

```python
# 1. Split by paragraphs
# 2. If paragraph > token limit, split by sentences
# 3. If sentence > token limit, split by tokens
# (Combine previous examples for implementation)
```

#### 6. LangChain's RecursiveCharacterTextSplitter

LangChain’s `RecursiveCharacterTextSplitter` splits text at natural boundaries, respects character limits, and preserves context with overlaps.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
LangChain is a powerful framework for building applications with language models.
It provides tools for chaining LLM calls, managing memory, and integrating with external data sources.
Chunking is essential when working with large documents to ensure compatibility with model input limits.
RecursiveCharacterTextSplitter helps split text at natural boundaries while respecting character limits.
"""

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,       # Max characters per chunk
    chunk_overlap=20      # Overlap between chunks to preserve context
)

chunks = text_splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n{'-'*40}")
```

**Output:**  
Each chunk is up to 100 characters, with 20-character overlaps, preserving context for downstream tasks.

**Best Practices:**

- Use token-based or semantic chunking for vector DBs and RAG.
- Always check your model’s max token limit.
- Prefer pretrained tokenizers for consistency.
- Recursive chunking is ideal for context preservation.
