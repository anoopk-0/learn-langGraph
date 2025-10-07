## Reciprocal Rank Fusion (RRF)

**Reciprocal Rank Fusion (RRF)** is a straightforward technique for merging ranked lists from multiple retrieval models (e.g., BM25, dense embeddings) into a single, improved ranking. It’s commonly used in Information Retrieval and Retrieval-Augmented Generation (RAG) pipelines to leverage the strengths of different retrievers.

- Different retrievers may rank documents differently.
- RRF rewards documents that appear in multiple lists and prioritizes those ranked higher.
- It’s simple, robust, and effective—often outperforming more complex fusion methods.

**How Does RRF Work?**

For each document d, the RRF score is calculated as:

    RRF(d) = sum over all ranked lists r of [1 / (k + rank_r(d))]

Where:

- R is the set of ranked lists from different retrievers.
- rank_r(d) is the position of document d in list r (1-based).
- k is a constant (commonly 60) that reduces the impact of rank position.

Key Points:

- Documents ranked higher get larger scores.
- Documents appearing in multiple lists accumulate more points.

```txt
- **Retriever A:** `[Doc1, Doc2, Doc3]`
- **Retriever B:** `[Doc3, Doc2, Doc4]`
- Let \( k = 60 \).

**Ranks:**
- Retriever A: Doc1=1, Doc2=2, Doc3=3
- Retriever B: Doc3=1, Doc2=2, Doc4=3

**RRF Scores:**
- Doc1: \( \frac{1}{61} \approx 0.0164 \)
- Doc2: \( \frac{1}{62} + \frac{1}{62} \approx 0.0323 \)
- Doc3: \( \frac{1}{63} + \frac{1}{61} \approx 0.0326 \)
- Doc4: \( \frac{1}{63} \approx 0.0159 \)

**Final Ranking:**
1. Doc3 (0.0326)
2. Doc2 (0.0323)
3. Doc1 (0.0164)
4. Doc4 (0.0159)

*Doc3 wins by appearing in both lists and ranking high.*

```

### Use Cases

- Hybrid search (BM25 + dense retrieval).
- RAG pipelines (merging multiple retrievers).
- Meta-search engines (aggregating results).
