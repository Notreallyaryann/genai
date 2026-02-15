from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import requests

load_dotenv()

def get_weather(city: str):
    url = f"http://wttr.in/{city}?format=%C+%t"  
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return f"The weather in {city} is {response.text.strip()}."
        else:
            return f"Could not get weather for {city}"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

# Initialize the model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.1,
    google_api_key=os.getenv("API_KEY")
)


city = "Kanpur"
weather_data = get_weather(city)
print(f"Weather data: {weather_data}")  

# Create prompt with weather data
prompt = f"Based on this weather information: '{weather_data}', give me a friendly weather report."
response = model.invoke(prompt)
print("\n" + "="*50)
print(response.content)