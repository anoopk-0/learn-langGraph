# hybrid_search.py
# Hybrid search combining vector similarity and BM25 keyword search for improved document retrieval.

from ingestion_pipline import vectorstore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# --- Initialize Vector Retriever ---
# Uses vector similarity to find documents most relevant to the query.
vector_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Return top 3 results
)

# --- Initialize BM25 Retriever ---
# Uses keyword-based BM25 algorithm for document retrieval.
documents = getattr(vectorstore, '_original_documents', None)
if not documents:
    raise RuntimeError(
        "Original documents not found in vectorstore. "
        "Please re-ingest or update ingestion_pipline.py."
    )
bm25_retriever = BM25Retriever.from_documents(documents, k=3)

# --- Create Hybrid (Ensemble) Retriever ---
# Combines vector and BM25 retrievers with specified weights.
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # 70% vector, 30% BM25
)

def run_search(query: str):
    """Runs the query through vector, BM25, and hybrid retrievers and prints results."""
    print(f"\nQuery: {query}\n")

    # Vector Search Results
    print("--- Vector Search Results ---")
    vector_results = vector_retriever.invoke(query)
    for i, doc in enumerate(vector_results, 1):
        print(f"Vector Document {i}:\n{doc.page_content}\n")

    # BM25 Search Results
    print("--- BM25 Search Results ---")
    bm25_results = bm25_retriever.invoke(query)
    for i, doc in enumerate(bm25_results, 1):
        print(f"BM25 Document {i}:\n{doc.page_content}\n")

    # Hybrid Search Results
    print("--- Hybrid Search Results (Weighted Ensemble) ---")
    hybrid_results = hybrid_retriever.invoke(query)
    for i, doc in enumerate(hybrid_results, 1):
        print(f"Hybrid Document {i}:\n{doc.page_content}\n")

if __name__ == "__main__":
    # Example query for demonstration
    query = "Provide a summary of the stock market performance in 2024."
    run_search(query)