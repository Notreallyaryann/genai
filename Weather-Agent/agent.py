from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
import json
import requests
import os

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

def run_command(cmd: str):
    result = os.system(cmd)
    return result

def get_weather(city: str):
    url = f"http://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."
    
    return "Something went wrong"


available_tools = {
    "get_weather": get_weather,
    "run_command": run_command
}

SYSTEM_PROMPT = f"""
    You are an helpfull AI Assistant who is specialized in resolving user query.
    You work on start, plan, action, observe mode.

    For the given user query and available tools, plan the step by step execution, based on the planning,
    select the relevant tool from the available tool. and based on the tool selection you perform an action to call the tool.

    Wait for the observation and based on the observation from the tool call resolve the user query.

    Rules:
    - Follow the Output JSON Format.
    - Always perform one step at a time and wait for next input
    - Carefully analyse the user query

    Output JSON Format:
    {{
        "step": "string",
        "content": "string",
        "function": "The name of function if the step is action",
        "input": "The input parameter for the function",
    }}

    Available Tools:
    - "get_weather": Takes a city name as an input and returns the current weather for the city
    - "run_command": Takes linux command as a string and executes the command and returns the output after executing it.

    Example:
    User Query: What is the weather of new york?
    Output: {{ "step": "plan", "content": "The user is interseted in weather data of new york" }}
    Output: {{ "step": "plan", "content": "From the available tools I should call get_weather" }}
    Output: {{ "step": "action", "function": "get_weather", "input": "new york" }}
    Output: {{ "step": "observe", "output": "12 Degree Cel" }}
    Output: {{ "step": "output", "content": "The weather for new york seems to be 12 degrees." }}

"""

messages = [
    { "role": "user", "content": SYSTEM_PROMPT + "\n\nRemember: Always respond with valid JSON format as specified above." }
]

while True:
    query = input("> ")
    messages.append({ "role": "user", "content": query })

    while True:
        # Prepare the conversation for Gemini
        conversation = ""
        for msg in messages:
            if msg["role"] == "user":
                conversation += f"User: {msg['content']}\n"
            else:
                conversation += f"Assistant: {msg['content']}\n"
        
        conversation += "Assistant: "

        response = model.generate_content(conversation)
        
        # Clean the response to ensure it's valid JSON
        response_text = response.text.strip()
        # Remove any markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        messages.append({ "role": "assistant", "content": response_text })
        
        try:
            parsed_response = json.loads(response_text)
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid JSON response from Gemini: {response_text}")
            break

        if parsed_response.get("step") == "plan":
            print(f"🧠: {parsed_response.get('content')}")
            continue

        if parsed_response.get("step") == "action":
            tool_name = parsed_response.get("function")
            tool_input = parsed_response.get("input")

            print(f"🛠️: Calling Tool:{tool_name} with input {tool_input}")

            if available_tools.get(tool_name) is not None:
                output = available_tools[tool_name](tool_input)
                messages.append({ "role": "user", "content": json.dumps({ "step": "observe", "output": output }) })
                continue
        
        if parsed_response.get("step") == "output":
            print(f"🤖: {parsed_response.get('content')}")
            break