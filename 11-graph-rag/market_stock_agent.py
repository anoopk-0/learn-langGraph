import asyncio
from typing import TypedDict, Sequence, Annotated
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from ingestion_pipline import vectorstore

SYSTEM_PROMPT = (
    "You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 "
    "based on the PDF document loaded into your knowledge base. Use the retriever tool available to answer "
    "questions about the stock market performance data. You can make multiple calls if needed. "
    "If you need to look up some information before asking a follow up question, you are allowed to do that! "
    "Please always cite the specific parts of the documents you use in your answers."
)

# --- Retriever Setup ---
vector_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

documents = getattr(vectorstore, '_original_documents', None)
if not documents:
    raise RuntimeError(
        "Original documents not found in vectorstore. "
        "Please re-ingest or update ingestion_pipline.py."
    )
bm25_retriever = BM25Retriever.from_documents(documents, k=3)

hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

def reciprocal_rank_fusion(results_lists, k=5, weight=60):
    scores = {}
    for results in results_lists:
        for rank, doc in enumerate(results):
            doc_id = getattr(doc, "id", None) or getattr(doc, "page_content", None)
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (weight + rank + 1)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    doc_map = {getattr(doc, "id", None) or getattr(doc, "page_content", None): doc for results in results_lists for doc in results}
    return [doc_map[doc_id] for doc_id, _ in sorted_docs[:k]]

@tool
def retriever_tool(query: str) -> str:
    """
    Searches and returns information from the Stock Market Performance 2024 document using both similarity and BM25 search, fused by RRF.
    """
    hybrid_fused_docs = hybrid_retriever.invoke(query)
    fused_docs = reciprocal_rank_fusion([hybrid_fused_docs], k=5)
    if not fused_docs:
        return "No relevant information found in the Stock Market Performance 2024 document."
    return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(fused_docs)])

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

tools = [retriever_tool]
llm = ChatOllama(model="llama3.2:3b", temperature=0, streaming=True).bind_tools(tools)

def llm_node(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": state["messages"] + [response]}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    return "continue" if getattr(last_message, "tool_calls", None) else "END"

graph = StateGraph(AgentState)
graph.add_node("llm_node", llm_node)
graph.add_node("tool_node", ToolNode(tools=tools))
graph.add_edge(START, "llm_node")
graph.add_conditional_edges("llm_node", should_continue, {
    "continue": "tool_node",
    "END": END
})
graph.add_edge("tool_node", "llm_node")
app = graph.compile()

async def run_agent():
    print("📊 Stock Market Q&A Agent Ready")
    print("Type 'exit' to quit.")
    while True:
        user_input = input("\n🔹 User: ")
        if user_input.lower() == "exit":
            break
        input_state = {"messages": [HumanMessage(content=user_input)]}
        print("Thinking...\n")
        async for event in app.astream_events(input=input_state, version="v2"):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk and chunk.content:
                    print(chunk.content, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(run_agent())
