import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# --- 1. INITIALIZATION ---
load_dotenv()
st.set_page_config(page_title="MaiStorage Agentic RAG", layout="wide", page_icon="🤖")

api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("🔑 API Key not found in Secrets or .env")
    st.stop()

# UPDATED MODELS FOR APRIL 2026
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
# Using the new 'gemini-embedding-001' which replaces the old 'text-embedding-004'
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    google_api_key=api_key
)

class AgentState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# --- 2. SIDEBAR UPLOADER ---
with st.sidebar:
    st.header("📂 Knowledge Management")
    uploaded_file = st.file_uploader("Upload Tech Specs (PDF)", type="pdf")
    
    if uploaded_file:
        with st.spinner("Indexing into Unified Embedding Space..."):
            with open("temp_k.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader("temp_k.pdf")
            docs = loader.load()
            
            # Recursive splitting to keep technical context intact
            # 
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            
            # Create the vector store
            st.session_state.vector_db = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings,
                collection_name="maistorage_internal"
            )
            st.success("✅ Knowledge Base Ready")

# --- 3. AGENTIC NODES ---
def retrieve_node(state):
    if "vector_db" not in st.session_state:
        return {"documents": []}
    docs = st.session_state.vector_db.similarity_search(state["question"], k=3)
    return {"documents": docs}

def generate_node(state):
    if not state["documents"]:
        return {"generation": "⚠️ Please upload a document to begin."}
    
    context = "\n\n".join([d.page_content for d in state["documents"]])
    prompt = f"Using ONLY this context: {context}\n\nQuestion: {state['question']}"
    
    response = llm.invoke(prompt)
    return {"generation": response.content}

# --- 4. ORCHESTRATION ---
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
agent = workflow.compile()

# --- 5. UI ---
st.title("🤖 MaiStorage Agentic RAG")

if "vector_db" not in st.session_state:
    st.info("Agent is offline. Please upload a PDF spec sheet to initialize.")

query = st.chat_input("Ask about aiDAPTIV+ or storage specs...")

if query:
    with st.chat_message("user"): st.write(query)
    with st.chat_message("assistant"):
        with st.status("Agent Reasoning..."):
            result = agent.invoke({"question": query})
        st.write(result["generation"])
