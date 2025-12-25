import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def ensure_directory_exists(path: str) -> bool:
    """Ensure the given directory exists, create if not."""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        print(f"❌ Failed to create directory '{path}': {e}")
        return False

def load_pdf_chunks(pdf_path: str, chunk_size: int = 1500, chunk_overlap: int = 250):
    """Load and split PDF into text chunks."""
    if not os.path.isfile(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return None
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        print(f"✅ PDF loaded. Pages: {len(pages)}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(pages)
        print(f"✅ Text chunks created: {len(chunks)}")
        return chunks
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return None

def create_chroma_vectorstore(chunks, persist_dir: str, embedding_model: str = "all-minilm"):
    """Create and persist Chroma vector store from text chunks."""
    import pickle
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model=embedding_model),
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print("✅ Chroma vector store initialized.")
        # Persist original chunks for BM25/hybrid search
        orig_docs_path = os.path.join(persist_dir, "original_documents.pkl")
        try:
            with open(orig_docs_path, "wb") as f:
                pickle.dump(chunks, f)
            print(f"✅ Original chunks persisted: {orig_docs_path}")
        except Exception as e:
            print(f"❌ Failed to persist original chunks: {e}")
        vectorstore._original_documents = chunks  # For retrieval
        return vectorstore
    except Exception as e:
        print(f"❌ Error initializing Chroma vector store: {e}")
        return None

def load_or_create_vectorstore(persist_dir: str, pdf_path: str):
    """Load existing Chroma vector store or create a new one from PDF."""
    import pickle
    # Try loading existing vector store
    if os.path.isdir(persist_dir):
        try:
            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=OllamaEmbeddings(model="all-minilm"),
                collection_metadata={"hnsw:space": "cosine"}
            )
            print("✅ Loaded existing Chroma vector store.")
            # Load original chunks for BM25/hybrid search
            orig_docs_path = os.path.join(persist_dir, "original_documents.pkl")
            if os.path.isfile(orig_docs_path):
                with open(orig_docs_path, "rb") as f:
                    vectorstore._original_documents = pickle.load(f)
                print(f"✅ Loaded original chunks: {orig_docs_path}")
            else:
                print(f"❌ original_documents.pkl not found. Re-ingestion required for BM25/hybrid search.")
                vectorstore._original_documents = None
            return vectorstore
        except Exception as e:
            print(f"ℹ️ Failed to load existing vector store: {e}\nCreating new one...")

    # Create new vector store if not found or failed to load
    if not ensure_directory_exists(persist_dir):
        return None
    chunks = load_pdf_chunks(pdf_path)
    if chunks is None:
        return None
    return create_chroma_vectorstore(chunks, persist_dir)

# Configuration
PERSIST_DIR = "./chroma_db"
PDF_PATH = "./store/Stock_Market_Performance_2024.pdf"

# Initialize vectorstore at module level
vectorstore = load_or_create_vectorstore(PERSIST_DIR, PDF_PATH)
if vectorstore is None:
    raise RuntimeError("Failed to initialize vectorstore.")
