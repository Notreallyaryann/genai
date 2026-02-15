from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
     model="models/gemini-embedding-001",
    google_api_key=os.getenv("API_KEY")
)

text = "dog chases cat"

vector = embeddings.embed_query(text)

print("Vector Embedding:", vector[:10]) 
print("Length of embedding:", len(vector))