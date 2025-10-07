## Tokens
A token is a unit of text used in natural language processing (NLP) and large language models (LLMs). Tokens can be words, subwords, or individual characters, depending on the tokenization method.

 - A whole word (e.g., "cat" → 1 token)
 - A part of a word (e.g., "running" → "run" + "ning" → 2 tokens)
 - Punctuation or spaces can also be separate tokens.

tokenization:
```txt
"Hello, world!" → ["Hello", ",", " world", "!"] → 4 tokens
```

Each model has a maximum token limit (context window) for processing input and generating output. If your input + output exceeds this limit, the model may truncate the response or fail to generate it properly.

 | Model    | Max Tokens         |
 |----------|-------------------|
 | GPT-4    | ~8,000 to 32,000  |
 | LLaMA    | ~8,000            |


- Tokens are the basic building blocks for LLMs. Models process and generate text at the token level.
- The number of tokens affects model input limits, cost, and performance.
- Tokenization helps models handle different languages, misspellings, and new words.

```py
from transformers import AutoTokenizer

# Use Meta's LLaMA 3 tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hello, how are you?"
tokens = tokenizer.encode(text, add_special_tokens=False)

print("Tokens:", tokens)
print("Token count:", len(tokens))
```