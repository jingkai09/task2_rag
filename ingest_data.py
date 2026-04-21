import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- CONFIGURATION ---
# Paste your NEW API key here. 
# WARNING: Do not upload this file to GitHub with the key visible!
API_KEY = "AIzaSyBpaYk9WSnjA8puTdlL-qSIglBX3Kj82NE"

def ingest():
    # 1. Load Data
    # The script looks for a file named 'sample.pdf' in a 'data' folder
    path = "data/sample.pdf"
    
    if os.path.exists(path):
        loader = PyPDFLoader(path)
    else:
        # Fallback for demo if PDF is missing
        print("⚠️ PDF not found at 'data/sample.pdf'. Creating a dummy text file for demo.")
        os.makedirs("data", exist_ok=True)
        with open("data/knowledge.txt", "w") as f:
            f.write("MaiStorage aiDAPTIV+ uses NAND flash to expand GPU VRAM for LLM training.")
        loader = TextLoader("data/knowledge.txt")
    
    documents = loader.load()

    # 2. Split (Recursive splitting is best for technical manuals)
    # Chunk size 2000 helps stay under the 15-request-per-minute Free Tier limit
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # 3. Embed and Store
    # Using the stable 'gemini-embedding-001' model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=API_KEY
    )
    
    # This creates a folder named 'chroma_db' where your data is stored
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    
    print(f"✅ Database created with {len(chunks)} chunks in ./chroma_db")

if __name__ == "__main__":
    ingest()
