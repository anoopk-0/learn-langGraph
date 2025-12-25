## Document Retriever

A **retriever** is a component in information retrieval systems that searches a collection of documents and returns those most relevant to a user's query. Retrievers use various algorithms to measure relevance, such as vector similarity, score thresholds, or diversity-based methods.

### Types of Retrievers

#### 1. Similarity Retriever

Returns documents that are most relevant to the query by comparing vector embeddings. Use when you want to retrieve the most relevant documents based solely on their similarity to the query, without additional filtering or diversity constraints.

```python
retriever_similarity = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

- `search_type="similarity"`: Uses vector similarity to rank documents.
- `search_kwargs={"k": 3}`: Retrieves the top 3 documents with the highest similarity scores.
```

#### 2. similarity_score_threshold

Returns documents that are similar to the query, but only if they meet a minimum similarity score. Use when you want to filter out less relevant results and ensure a minimum relevance threshold.

```python
retriever_similarity_threshold = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.3}
)

- `search_type="similarity_score_threshold"`: Uses vector similarity and applies a score threshold.
- `search_kwargs={"k": 3, "score_threshold": 0.3}`: Retrieves up to 3 documents, but only those with a similarity score of at least 0.3.
```

#### 3. Max Marginal Relevance (MMR)

Returns documents that are both relevant to the query and diverse, reducing redundancy in the results. Use when you want to ensure the retrieved documents cover different aspects of the query and avoid duplicates.

**When not to use MMR:**  
Avoid using MMR if you only care about retrieving the most relevant documents and do not need diversity, or if your dataset is small and redundancy is not a concern. MMR may also be less suitable when speed is critical, as it can be slower than pure similarity search.

```python
retriever_max_marginal_relevance = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
)

- `search_type="mmr"`: Uses Max Marginal Relevance to balance relevance and diversity.
- `search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}`:
    - `k`: Number of documents to return.
    - `fetch_k`: Number of documents initially considered for diversity.
    - `lambda_mult`: Controls the balance between relevance and diversity (0.5 gives equal weight). 0 = max diversity , 1 = max relevance, 0.5 is in the middle
```

```python
# Retrieve documents about stock market performance in 2024 using MMR retriever
result = retriever_max_marginal_relevance.invoke("What is the stock market performance in 2024?")

for i, doc in enumerate(result):
        print(f"Document {i+1}:\n{doc.page_content}\n")
```
