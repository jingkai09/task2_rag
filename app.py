import streamlit as st
import os
import time
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

# Secure API Key loading
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("🔑 API Key not found. Please add GOOGLE_API_KEY to Streamlit Secrets or .env file.")
    st.stop()

# Using Gemini 1.5 Flash and the latest stable Embedding model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", # Updated to the latest stable ID
    google_api_key=api_key
)

# Define the state for our Agentic Loop
class AgentState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# --- 2. SIDEBAR: DATA INGESTION ---
with st.sidebar:
    st.header("📂 Knowledge Management")
    st.write("Upload technical docs to initialize the Agent's memory.")
    uploaded_file = st.file_uploader("Upload MaiStorage Tech Specs (PDF)", type="pdf")
    
    if uploaded_file:
        # Check if the database is already built in this session to avoid re-indexing
        if "vector_db" not in st.session_state:
            with st.spinner("Indexing PDF... (Avoiding Rate Limits)"):
                # Save temporary file
                with open("temp_knowledge.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                loader = PyPDFLoader("temp_knowledge.pdf")
                docs = loader.load()
                
                # OPTIMIZATION: Larger chunks (2000) to stay under 100 API requests/min
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000, 
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ".", " "]
                )
                chunks = text_splitter.split_documents(docs)
                
                try:
                    # Create Vector Store
                    st.session_state.vector_db = Chroma.from_documents(
                        documents=chunks, 
                        embedding=embeddings,
                        collection_name="maistorage_internal"
                    )
                    st.success(f"✅ Indexed {len(chunks)} technical chunks.")
                except Exception as e:
                    st.error(f"Error during indexing: {e}")
                    st.info("The API might be rate-limited. Please wait 30 seconds and refresh.")

# --- 3. AGENTIC NODES (The Reasoning Logic) ---

def retrieve_node(state: AgentState):
    """Retrieval step with safety check."""
    if "vector_db" not in st.session_state:
        return {"documents": []}
    
    # Retrieve top 3 relevant chunks based on similarity
    docs = st.session_state.vector_db.similarity_search(state["question"], k=3)
    return {"documents": docs}

def generate_node(state: AgentState):
    """Generation step with grounded context."""
    if not state["documents"]:
        return {"generation": "⚠️ I don't have any internal data for this. Please upload a PDF in the sidebar."}
    
    # Join context for the LLM
    context = "\n\n".join([d.page_content for d in state["documents"]])
    
    prompt = f"""You are a MaiStorage Technical Assistant. 
    Use the following technical context to answer the question. 
    If the answer isn't in the context, politely say you don't have that specific data.
    
    CONTEXT:
    {context}
    
    QUESTION: {state['question']}"""
    
    response = llm.invoke(prompt)
    return {"generation": response.content}

# --- 4. GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
agent_app = workflow.compile()

# --- 5. CHAT INTERFACE ---
st.title("🤖 MaiStorage Agentic RAG")
st.caption("Technical Prototype for Internal Knowledge Retrieval")

if "vector_db" not in st.session_state:
    st.info("👈 Please upload a PDF spec sheet in the sidebar to begin.")

query = st.chat_input("Ask about storage architecture or specifications...")

if query:
    with st.chat_message("user"):
        st.write(query)
    
    with st.chat_message("assistant"):
        with st.status("Agent Reasoning...", expanded=False) as status:
            result = agent_app.invoke({"question": query})
            status.update(label="Response Ready", state="complete")
        
        st.markdown(result["generation"])
