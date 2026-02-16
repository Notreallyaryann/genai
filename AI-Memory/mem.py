from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_community.vectorstores import Qdrant
from langchain_neo4j import Neo4jGraph
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import os
from dotenv import load_dotenv


load_dotenv()


GEMINI_API_KEY = os.getenv("API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("API_KEY not found in .env file!")


QUADRANT_HOST = "localhost"
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "reform-william-center-vibrate-press-5829"


llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7
)


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)

graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

qdrant_client = QdrantClient(host=QUADRANT_HOST, port=6333)

collection_name = "memories"


try:
    qdrant_client.get_collection(collection_name)
    print(f" Collection '{collection_name}' exists.")
except Exception:
    print(f"🔄 Creating collection '{collection_name}'...")
    
    qdrant_client.create_collection(
        collection_name=collection_name,
       vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )
    print(f" Collection '{collection_name}' created.")


vector_store = Qdrant(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=embeddings,
)
print(" Qdrant vector store is ready!")


memory = ConversationBufferMemory(return_messages=True)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)


def store_in_neo4j(user_message, bot_response):
    """Store conversation in Neo4j graph"""
    query = """
    MERGE (s:Session {id: $session_id})
    CREATE (um:Message:UserMessage {
        content: $user_message,
        timestamp: datetime(),
        role: 'user'
    })
    CREATE (bm:Message:BotMessage {
        content: $bot_response,
        timestamp: datetime(),
        role: 'assistant'
    })
    CREATE (s)-[:HAS_MESSAGE]->(um)
    CREATE (s)-[:HAS_MESSAGE]->(bm)
    CREATE (um)-[:RESPONDED_WITH]->(bm)
    CREATE (bm)-[:RESPONDS_TO]->(um)
    """
    graph.query(query, {
        "session_id": "p123",
        "user_message": user_message,
        "bot_response": bot_response
    })

def search_similar_memories(query):
    """Search for similar past conversations in Qdrant"""
    results = vector_store.similarity_search(query, k=3)
    memories = []
    for doc in results:
        memories.append(doc.page_content)
    return "\n".join(memories)

def chat(message):
    print(f"\n🔍 Processing: {message}")
    
   
    similar_memories = search_similar_memories(message)
    

    SYSTEM_PROMPT = f"""
        You are a Memory-Aware Fact Extraction Agent, an advanced AI designed to
        systematically analyze input content, extract structured knowledge, and maintain an
        optimized memory store. Your primary function is information distillation
        and knowledge preservation with contextual awareness.

        Tone: Professional analytical, precision-focused, with clear uncertainty signaling
        
        Similar Past Conversations:
        {similar_memories if similar_memories else "No similar memories found."}
    """
    
   
    response = conversation.predict(input=message)
    
   
    store_in_neo4j(message, response)
    
    
    vector_store.add_texts(
        texts=[f"User: {message}\nAssistant: {response}"],
        metadatas=[{"session": "p123", "timestamp": "now"}]
    )
    
    return response



print("\n" + "="*60)
print("="*60)
print("Commands:")
print("  • Type message to chat")
print("  • '/exit' - Quit")
print("="*60)


while True:
    message = input(">> ")
    print("BOT: ", chat(message=message))
