import os
import time
import asyncio
from typing import TypedDict, Sequence, Annotated
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from sentence_transformers import CrossEncoder

PDF_FILENAME = "Stock_Market_Performance_2024.pdf"
EMBED_MODEL = "all-minilm"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 250
SYSTEM_PROMPT = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

# PDF Loading & Chunking
pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", PDF_FILENAME))
if not os.path.isfile(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)
pages = pdf_loader.load()
print(f"✅ PDF loaded. Pages: {len(pages)}")

# Split the loaded PDF pages into smaller text chunks for embedding and retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = text_splitter.split_documents(pages)
print(f"✅ Text chunks created: {len(chunks)}")

# Vector Store Setup
persist_directory = os.path.join(os.path.dirname(__file__), "stock_market_vector_store")
os.makedirs(persist_directory, exist_ok=True)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model=EMBED_MODEL),
    persist_directory=persist_directory
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})
print("✅ Chroma vector store initialized.")

# Reranker Setup
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(RERANK_MODEL)

def rerank(query, docs, top_k=3):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    # Sort docs by score descending
    sorted_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return sorted_docs[:top_k]

# Tool Definition
@tool
def retriever_tool(query: str) -> str:
    """
    Tool: Searches and returns information from the Stock Market Performance 2024 document, with reranking.
    Args:
        query (str): The search query.
    Returns:
        str: Relevant document chunks or a not found message.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the Stock Market Performance 2024 document."
    reranked_docs = rerank(query, docs, top_k=3)
    return "\n\n".join([f"Document {i+1} (reranked):\n{doc.page_content}" for i, doc in enumerate(reranked_docs)])

# Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# LangGraph Nodes
tools = [retriever_tool]
llm = ChatOllama(model="llama3.2:3b", temperature=0, streaming=True).bind_tools(tools)

def llm_node(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": state["messages"] + [response]}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    return "continue" if getattr(last_message, "tool_calls", None) else "END"

# LangGraph Construction
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

def type_out(text: str, delay: float = 0.01):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

async def run_agent():
    print("📊 Stock Market Q&A Agent Ready (with Reranking)")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\n🔹 User: ")
        if user_input.lower() == "exit":
            break

        input_state = {"messages": [HumanMessage(content=user_input)]}
        print("Thinking...\n")
        async for event in app.astream_events(input=input_state, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk and chunk.content:
                    print(chunk.content, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(run_agent())
