from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Literal
from pydantic import BaseModel
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

load_dotenv()


genai.configure(api_key=os.getenv("API_KEY"))

# Schema
class DetectCallResponse(BaseModel):
    is_question_ai: bool

class CodingAIResponse(BaseModel):
    answer: str

class State(TypedDict):
    user_message: str
    ai_message: str
    is_coding_question: bool

def detect_query(state: State):
    user_message = state.get("user_message")

    SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to detect if the user's query is related
    to coding question or not.
    Return the response in specified JSON boolean only.
    """

    # Gemini Call - using gemini-flash-latest
    model = genai.GenerativeModel('models/gemini-flash-latest')
    response = model.generate_content(
        f"""{SYSTEM_PROMPT}
        
        User query: {user_message}
        
        Return ONLY a valid JSON object in this format: {{"is_question_ai": true/false}}
        No other text, just the JSON."""
    )
    
    response_text = response.text.strip()
    if response_text.startswith('```json'):
        response_text = response_text.replace('```json', '').replace('```', '')
    elif response_text.startswith('```'):
        response_text = response_text.replace('```', '')
    
    result = json.loads(response_text.strip())
    state["is_coding_question"] = result.get("is_question_ai", False)
    return state

def route_edge(state: State) -> Literal["solve_coding_question", "solve_simple_question"]:
    is_coding_question = state.get("is_coding_question")

    if is_coding_question:
        return "solve_coding_question"
    else:
        return "solve_simple_question"

def solve_coding_question(state: State):
    user_message = state.get("user_message")

    # Gemini Call for coding questions
    SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to resolve the user query based on coding 
    problem he is facing
    """

    model = genai.GenerativeModel('models/gemini-2.5-flash')
    response = model.generate_content(
        f"""{SYSTEM_PROMPT}
        
        User's coding question: {user_message}
        
        Please provide a helpful response.
        """
    )
    
    state["ai_message"] = response.text
    return state

def solve_simple_question(state: State):
    user_message = state.get("user_message")

    # Gemini Call for simple questions - using gemini-flash-latest
    SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to chat with user
    """

    model = genai.GenerativeModel('models/gemini-flash-latest')
    response = model.generate_content(
        f"""{SYSTEM_PROMPT}
        
        User's question: {user_message}
        
        Please provide a friendly response.
        """
    )
    
    state["ai_message"] = response.text
    return state

graph_builder = StateGraph(State)

graph_builder.add_node("detect_query", detect_query)
graph_builder.add_node("solve_coding_question", solve_coding_question)
graph_builder.add_node("solve_simple_question", solve_simple_question)

graph_builder.add_edge(START, "detect_query")
graph_builder.add_conditional_edges("detect_query", route_edge)
graph_builder.add_edge("solve_coding_question", END)
graph_builder.add_edge("solve_simple_question", END)

graph = graph_builder.compile()

# Use the Graph
def call_graph():
    state = {
        "user_message": "Hello jii",
        "ai_message": "",
        "is_coding_question": False
    }
    
    result = graph.invoke(state)
    print("Final Result", result)

call_graph()