from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
import os
from dotenv import load_dotenv

load_dotenv()


GEMINI_API_KEY = os.getenv("API_KEY") 

QUADRANT_HOST = "localhost"
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "reform-william-center-vibrate-press-5829"


llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",  
    google_api_key=GEMINI_API_KEY,
    temperature=0.7
)


memory = ConversationBufferMemory(  
    return_messages=True
)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

def chat(message):
    print(f"\n🔍 Processing: {message}")
    
    response = conversation.predict(input=message)
    
    return response

print("\n" + "="*50)
print("🤖 Memory Chat")
print("="*50)
print("Type 'exit' to quit\n")



while True:
    message = input(">> ")
    print("BOT: ", chat(message=message))

