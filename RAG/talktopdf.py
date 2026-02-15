from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
from langchain_qdrant import QdrantVectorStore
import os

load_dotenv()

pdf_path = Path(__file__).parent / "Resume.pdf"


loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load() 
print(f" Loaded {len(docs)} pages")

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)

split_docs = text_splitter.split_documents(documents=docs)
print(f" Created {len(split_docs)} chunks")

# Vector Embeddings -
embedding_model = GoogleGenerativeAIEmbeddings(
   model="models/gemini-embedding-001",
    google_api_key=os.getenv("API_KEY")  
)


vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url="http://localhost:6333",
    collection_name="learning_vectors",
    embedding=embedding_model
)

print(" Indexing of Documents Done...")
print(f" Collection 'learning_vectors' created with {len(split_docs)} vectors")

#to chat

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    google_api_key=os.getenv("API_KEY")
)

print("\n" + "="*50)
print(" Now You can ask questions related to PDF")
print("="*50)

def ask_question(question):
    print(f"\nYour Question: {question}")
    
    results = vector_store.similarity_search(question, k=3)
    
    context = "\n\n".join([doc.page_content for doc in results])
    
  
    prompt = f"""Context diya gaya hai:
{context}

Question: {question}

Sirf context mein di gayi information se jawab do. Agar jawab nahi hai to batao ki nahi pata.
Hinglish mein jawab do (Hindi + English mix).
"""
    
    response = llm.invoke(prompt)
    
    print(f"🤖 Answer: {response.content}")
    return response.content


while True:
    user_input = input("\n❓ Aapka sawaal (exit likh kar bahar nikle): ")
    
    if user_input.lower() in ['exit', 'quit', 'bye', 'bahar', 'khatam']:
        print("👋 Alvida! Fir milenge!")
        break
    
    if user_input.strip():
        ask_question(user_input)