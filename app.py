import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# --- 1. INITIALIZATION & SECURITY ---
load_dotenv()
st.set_page_config(page_title="MaiStorage Agentic RAG", layout="wide", page_icon="🤖")

# Priority for API Key: Streamlit Secrets (Cloud) -> .env file (Local)
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("🔑 API Key not found. Please add GOOGLE_API_KEY to Streamlit Secrets or .env file.")
    st.stop()

# Initialize Gemini 1.5 Flash
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# --- 2. AGENT STATE DEFINITION ---
class AgentState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# --- 3. SIDEBAR: KNOWLEDGE MANAGEMENT ---
with st.sidebar:
    st.header("📂 Knowledge Management")
    st.write("Upload technical documents to build the agent's memory.")
    uploaded_file = st.file_uploader("Upload MaiStorage Tech Specs (PDF)", type="pdf")
    
    if uploaded_file:
        with st.spinner("Processing PDF and building Vector Store..."):
            # Save temp file
            with open("temp_knowledge.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load & Split
            loader = PyPDFLoader("temp_knowledge.pdf")
            docs = loader.load()
            
            # Recursive splitting maintains technical document structure
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(docs)
            
            # Store in Session State to keep it alive during the session
            st.session_state.vector_db = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings
            )
            st.success("✅ Knowledge Base Updated!")

# --- 4. AGENT NODES (The Reasoning Logic) ---

def retrieve_node(state: AgentState):
    """Retrieve chunks from the uploaded knowledge base."""
    if "vector_db" not in st.session_state:
        return {"documents": []}
    
    # Retrieve top 3 relevant chunks
    docs = st.session_state.vector_db.similarity_search(state["question"], k=3)
    return {"documents": docs}

def generate_node(state: AgentState):
    """Grounded generation with a 'No-Data' safety check."""
    if not state["documents"]:
        return {"generation": "⚠️ I don't have any internal data regarding this query. Please upload a relevant PDF in the sidebar."}
    
    # Format context with source metadata for citations
    context = "\n\n".join([f"[Source: {d.metadata.get('source', 'PDF')}] {d.page_content}" for d in state["documents"]])
    
    prompt = f"""You are a MaiStorage Technical Assistant. 
    Using the context provided below, answer the user's question. 
    Strictly avoid hallucinations. If the answer is not in the context, say you don't know.
    
    CONTEXT:
    {context}
    
    QUESTION: {state['question']}"""
    
    response = llm.invoke(prompt)
    return {"generation": response.content}

# --- 5. LANGGRAPH ORCHESTRATION ---
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
agent_app = workflow.compile()

# --- 6. CHAT INTERFACE ---
st.title("🤖 MaiStorage Agentic RAG")
st.markdown("---")

# Instruction for the interviewer
if "vector_db" not in st.session_state:
    st.warning("Please upload a PDF in the sidebar to begin the Agentic RAG demo.")

query = st.chat_input("Ask a technical question about internal specs...")

if query:
    # Display user message
    with st.chat_message("user"):
        st.write(query)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.status("Agent Reasoning...", expanded=True) as status:
            result = agent_app.invoke({"question": query})
            status.update(label="Response Generated", state="complete")
        
        st.markdown(result["generation"])
