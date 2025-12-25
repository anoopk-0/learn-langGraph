from langchain_core.tools import tool
from langchain_mcp_adapters.tools import to_fastmcp
from mcp.server.fastmcp import FastMCP
import requests

@tool
def call_openapi(endpoint: str, method: str = "GET", params: dict = None, data: dict = None) -> str:
    """Call an OpenAPI endpoint with the given method, params, and data."""
    try:
        response = requests.request(method, endpoint, params=params, json=data)
        return response.text
    except Exception as e:
        return f"OpenAPI error: {e}"

@tool
def get_weather(city: str) -> str:
    """Get the current temperature in Celsius for a given city using Open-Meteo API."""
    try:
        geo_resp = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}")
        geo = geo_resp.json()
        if not geo.get("results"):
            return f"City '{city}' not found."
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]
        weather_resp = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        )
        weather = weather_resp.json()
        temp = weather["current_weather"]["temperature"]
        return f"The current temperature in {city} is {temp}°C."
    except Exception as e:
        return f"Error fetching weather: {e}"

@tool
def get_users() -> str:
    """Fetch 10 users from JSONPlaceholder API."""
    try:
        users_resp = requests.get("https://jsonplaceholder.typicode.com/users")
        users = users_resp.json()
        return "\n".join([f"{u['id']}: {u['name']} ({u['email']})" for u in users[:10]])
    except Exception as e:
        return f"Error fetching users: {e}"

fastmcp_tools = [to_fastmcp(call_openapi), to_fastmcp(get_weather), to_fastmcp(get_users)]
mcp = FastMCP("OpenAPI", tools=fastmcp_tools)
mcp.run(transport="stdio")
