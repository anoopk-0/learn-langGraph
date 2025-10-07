from ingestion_pipline import vectorstore
from typing import TypedDict, List, Dict, Tuple, Hashable
from langchain_ollama import ChatOllama

class AgentState(TypedDict):
    queryList: list[str]

llm = ChatOllama(model="llama3.2:3b", temperature=0, streaming=True)
original_query = "Provide a summary of the stock market performance in 2024."

# NOTE:
# with_structured_output: This method wraps the LLM to ensure its output conforms to the specified TypedDict structure.
# It helps in extracting structured data from the LLM's response.
llm_with_tool = llm.with_structured_output(AgentState)

prompt = (
    "Generate three different variations of this query for document retrieval:\n"
    f"original query: {original_query}\n"
    "Ensure each query is concise and highlights a different aspect."
)

response = llm_with_tool.invoke(prompt)
query_variations = response["queryList"]

retrieved_docs_lists = []
for idx, query in enumerate(query_variations, start=1):
    print(f"Query {idx}: {query}")
    docs = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    ).invoke(query)
    doc_ids = [doc.page_content for doc in docs]
    retrieved_docs_lists.append(doc_ids)

# NOTE: 
# RRF (Reciprocal Rank Fusion) is a method to combine multiple ranked lists into a single ranking.
# It assigns scores to documents based on their ranks in each list, favoring documents that appear, more frequently across the lists.
def rrf_fuse(
    rankings: List[List[Hashable]],
    k: int = 60
) -> List[Tuple[Hashable, float]]:
    scores: Dict[Hashable, float] = {}
    for ranked_list in rankings:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: (-x[1], str(x[0])))
    return fused

fused_docs = rrf_fuse(retrieved_docs_lists)

print("\nFused documents (by RRF):")
for doc_id, score in fused_docs:
    print(f"Doc ID: {doc_id}, RRF Score: {score:.4f}")
