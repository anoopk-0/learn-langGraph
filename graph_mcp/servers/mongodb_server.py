from langchain_core.tools import tool
from langchain_mcp_adapters.tools import to_fastmcp
from mcp.server.fastmcp import FastMCP
from pymongo import MongoClient
import os

# MongoDB connection configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGO_DB", "langgraph_db")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "product_collection")

@tool
def find_documents(query: dict) -> str:
    """Return documents matching the query from MongoDB."""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        results = list(collection.find(query))
        return str(results)
    except Exception as e:
        return f"MongoDB error: {e}"

@tool
def get_all_products() -> str:
    """Return all products from the MongoDB collection."""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        results = list(collection.find())
        return str(results)
    except Exception as e:
        return f"MongoDB error: {e}"

@tool
def get_product_by_id(product_id: str) -> str:
    """Return details for a single product by its ID."""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        result = collection.find_one({"_id": product_id})
        if result:
            return str(result)
        else:
            return f"Product with ID {product_id} not found."
    except Exception as e:
        return f"MongoDB error: {e}"
    
@tool
def create_and_save_product(product_data: dict) -> str:
    """Create and save a new product to the MongoDB collection."""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        result = collection.insert_one(product_data)
        return f"Product created with ID: {result.inserted_id}"
    except Exception as e:
        return f"MongoDB error: {e}"

# Register MongoDB tools for FastMCP
fastmcp_tools = [
    to_fastmcp(find_documents),
    to_fastmcp(get_all_products),
    to_fastmcp(get_product_by_id),
    to_fastmcp(create_and_save_product)
]

# Create and run the MCP server using stdio transport
mcp = FastMCP("MongoDB", tools=fastmcp_tools)
mcp.run(transport="stdio")
