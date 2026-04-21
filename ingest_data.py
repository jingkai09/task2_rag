import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

def ingest():
    # 1. Load Data (Change 'knowledge.txt' to your actual file)
    if os.path.exists("data/sample.pdf"):
        loader = PyPDFLoader("data/sample.pdf")
    else:
        # Fallback to a simple text file if no PDF exists
        with open("data/knowledge.txt", "w") as f:
            f.write("MaiStorage aiDAPTIV+ uses NAND flash to expand GPU VRAM.")
        loader = TextLoader("data/knowledge.txt")
    
    documents = loader.load()

    # 2. Split (Recursive is better for technical docs)
    # 
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # 3. Embed and Store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    print("✅ Database created in ./chroma_db")

if __name__ == "__main__":
    ingest()
