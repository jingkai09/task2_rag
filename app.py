import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings, HarmBlockThreshold, HarmCategory
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
    st.error("🔑 API Key not found. Please add it to Streamlit Secrets.")
    st.stop()

# Initialize Chat with Safety Filters turned OFF (to avoid blocking technical specs)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=api_key,
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }
)

# Initialize Embeddings with the 2026 stable model name
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    google_api_key=api_key
)

class AgentState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# --- 2. SIDEBAR: KNOWLEDGE MANAGEMENT ---
with st.sidebar:
    st.header("📂 Knowledge Management")
    uploaded_file = st.file_uploader("Upload Tech Specs (PDF)", type="pdf")
    
    if uploaded_file:
        if "vector_db" not in st.session_state:
            with st.spinner("Processing technical document..."):
                with open("temp_k.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                loader = PyPDFLoader("temp_k.pdf")
                docs = loader.load()
                
                # Split into larger 2000-char chunks to reduce API calls
                splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
                
                # Create vector store
                st.session_state.vector_db = Chroma.from_documents(
                    documents=chunks, 
                    embedding=embeddings,
                    collection_name="maistorage_internal"
                )
                st.success(f"✅ Knowledge Base Ready")

# --- 3. AGENTIC NODES ---
def retrieve_node(state: AgentState):
    if "vector_db" not in st.session_state:
        return {"documents": []}
    # Retrieve top 2 chunks to keep the prompt size manageable for the free tier
    docs = st.session_state.vector_db.similarity_search(state["question"], k=2)
    return {"documents": docs}

def generate_node(state: AgentState):
    if not state["documents"]:
        return {"generation": "⚠️ Please upload a PDF in the sidebar first."}
    
    # Grounded Context
    context = "\n\n".join([d.page_content for d in state["documents"]])
    prompt = f"Using ONLY the following context, answer the question: {state['question']}\n\nCONTEXT:\n{context}"
    
    try:
        response = llm.invoke(prompt)
        return {"generation": response.content}
    except Exception as e:
        return {"generation": "🛰️ Rate Limit: The AI is busy. Please wait 15 seconds and try again!"}

# --- 4. ORCHESTRATION ---
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
agent_app = workflow.compile()

# --- 5. UI ---
st.title("🤖 MaiStorage Agentic RAG")
query = st.chat_input("Ask a technical question...")

if query:
    with st.chat_message("user"): st.write(query)
    with st.chat_message("assistant"):
        with st.status("Agent Reasoning..."):
            result = agent_app.invoke({"question": query})
        st.write(result["generation"])
