from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def create_db():
    # 1. Load your local knowledge (e.g., a text file of MaiStorage's aiDAPTIV+ specs)
    loader = TextLoader("sample_data/knowledge.txt")
    docs = loader.load()

    # 2. Split into chunks (Recursive avoids cutting technical specs in half)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # 3. Embed and Store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    print("Vector database created successfully.")

if __name__ == "__main__":
    create_db()
