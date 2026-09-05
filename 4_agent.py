import os
import requests

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.tools import tool


# =========================
# Load environment variables
# =========================

load_dotenv()


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "your_langsmith_project"



# =========================
# Search Tool
# =========================

search_tool = TavilySearch(
    max_results=5
)


# =========================
# Weather Tool
# =========================

@tool
def get_weather_data(city: str) -> str:
    """
    Get current weather for a city using Open-Meteo.
    Open-Meteo does NOT require an API key.
    """

    # Geocoding
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"

    geo_params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    geo_response = requests.get(
        geo_url,
        params=geo_params
    )

    geo_data = geo_response.json()

    if "results" not in geo_data:
        return f"Could not find city: {city}"

    location = geo_data["results"][0]

    latitude = location["latitude"]
    longitude = location["longitude"]
    city_name = location["name"]
    country = location.get("country", "")

    # Current weather
    weather_url = "https://api.open-meteo.com/v1/forecast"

    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "temperature_unit": "celsius"
    }

    weather_response = requests.get(
        weather_url,
        params=weather_params
    )

    weather_data = weather_response.json()

    current = weather_data["current"]

    return (
        f"City: {city_name}, {country}\n"
        f"Temperature: {current['temperature_2m']}°C\n"
        f"Humidity: {current['relative_humidity_2m']}%\n"
        f"Wind Speed: {current['wind_speed_10m']} km/h"
    )

# =========================
# ChatGroq LLM
# =========================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# =========================
# Create Agent
# =========================

agent = create_agent(
    model=llm,
    tools=[
        search_tool,
        get_weather_data
    ],
    system_prompt="""
    You are a helpful AI assistant.

    You have two tools:

    1. TavilySearch:
       Use it for web searches and current information.

    2. get_weather_data:
       Use it for weather questions.
       This tool uses Open-Meteo.
       Open-Meteo does NOT require an API key.

    NEVER ask the user for a weather API key.
    Always use get_weather_data for weather questions.

    Give clear and concise answers.
    """
)


# =========================
# Run Agent
# =========================

while True:

    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        }
    )

    print("\nAI:", response["messages"][-1].content)