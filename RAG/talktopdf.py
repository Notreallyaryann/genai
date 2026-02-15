from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient  
import os

load_dotenv()

pdf_path = Path(__file__).parent / "Resume.pdf"

#  Check if collection exists
client = QdrantClient(url="http://localhost:6333")
collection_name = "learning_vectors"

# Get all collections
collections = client.get_collections().collections
collection_names = [c.name for c in collections]

if collection_name not in collection_names:
    #  create karo
    print("Loading PDF...")
    
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

    # Vector Embeddings
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("API_KEY")  
    )

    vector_store = QdrantVectorStore.from_documents(
        documents=split_docs,
        url="http://localhost:6333",
        collection_name=collection_name,
        embedding=embedding_model
    )

    print(" Indexing of Documents Done...")
    print(f" Collection '{collection_name}' created with {len(split_docs)} vectors")
    
else:
    #  sirf connect karo
    print("✅ Collection already exists! Connecting to existing vectors...")
    
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("API_KEY")  
    )
    
    vector_store = QdrantVectorStore(
        client=client,  
        collection_name=collection_name,
        embedding=embedding_model
    )
    
   
    collection_info = client.get_collection(collection_name)
    print(f"📊 Existing collection is ready")


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