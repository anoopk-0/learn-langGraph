## BM25

BM25 (Best Matching 25) is a ranking function used by search engines to estimate the relevance of documents to a given search query. It is based on the probabilistic retrieval framework and improves upon earlier models like TF-IDF by introducing term saturation and document length normalization.

- **Term Frequency (TF):** Measures how often a term appears in a document. BM25 uses a saturation function so that the impact of term frequency increases but with diminishing returns.
- **Inverse Document Frequency (IDF):** Measures how unique or rare a term is across all documents. Rare terms are given more weight.
- **Document Length Normalization:** Adjusts scores so longer documents are not unfairly favored.

#### BM25 Formula

For a query Q with terms q₁, q₂, ..., qₙ, the BM25 score for a document D is calculated as follows:

For each query term qᵢ:
- Compute the term frequency f(qᵢ, D): how many times qᵢ appears in D.
- Compute the inverse document frequency IDF(qᵢ): gives higher weight to rare terms.
- Adjust for document length |D| and average document length avgdl.

The BM25 score is the sum over all query terms:

score(D, Q) = sum over i of [ IDF(qᵢ) × (f(qᵢ, D) × (k₁ + 1)) / (f(qᵢ, D) + k₁ × (1 - b + b × (|D| / avgdl))) ]

Where:
- f(qᵢ, D): Frequency of term qᵢ in document D
- |D|: Length of document D
- avgdl: Average document length in the corpus
- k₁: Controls term frequency saturation (typically 1.2–2.0)
- b: Controls document length normalization (typically 0.75)
- IDF(qᵢ): Inverse document frequency of term qᵢ

### Example

Suppose you have the following documents:

- **Doc1:** "BM25 is a ranking function"
- **Doc2:** "BM25 improves TF-IDF"
- **Doc3:** "TF-IDF is a classic model"

Query: `"BM25 ranking"`

Assume:
- \( k_1 = 1.5 \), \( b = 0.75 \)
- Average document length = 5 words

Calculate BM25 scores for each document:

1. **Term Frequencies:**
    - "BM25" appears in Doc1 and Doc2.
    - "ranking" appears in Doc1.

2. **IDF Calculation:**
    - "BM25" appears in 2/3 documents: \( \text{IDF} = \log\left(\frac{3 - 2 + 0.5}{2 + 0.5}\right) \)
    - "ranking" appears in 1/3 documents: \( \text{IDF} = \log\left(\frac{3 - 1 + 0.5}{1 + 0.5}\right) \)

3. **Score Calculation:**
    - For Doc1, both terms are present, so it will have the highest score.
    - For Doc2, only "BM25" is present.
    - For Doc3, neither term is present, so score is zero.

```py
from langchain.retrievers import BM25Retriever

# Sample documents
docs = [
    Document(page_content="Python is a programming language."),
    Document(page_content="LangChain helps build LLM applications."),
    Document(page_content="BM25 is a ranking function used in information retrieval."),
]

# Create BM25 retriever
retriever = BM25Retriever.from_documents(docs)
retriever.k = 2  # Number of top documents to retrieve

# Define node: retrieve documents
def retrieve_node(state: GraphState) -> GraphState:
    query = state["query"]
    results = retriever.get_relevant_documents(query)
    return GraphState({**state, "retrieved_docs": results})

```