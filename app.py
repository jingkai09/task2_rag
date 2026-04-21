import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. CONFIGURATION ---
# Paste your NEW API Key here for the demo
API_KEY = "PASTE_YOUR_NEW_API_KEY_HERE" 

st.set_page_config(page_title="MaiStorage AI Assistant", layout="wide")

# Initialize Chat: Gemini 2.5 Flash is currently the top choice for reasoning
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=API_KEY,
    temperature=0.1
)

# Initialize Embedding: 'gemini-embedding-001' is the current STABLE standard
# Alternatively, 'gemini-embedding-2-preview' for advanced multimodal RAG
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    google_api_key=API_KEY
)

# --- 2. SIDEBAR: DOCUMENT INGESTION ---
with st.sidebar:
    st.header("📂 Data Ingestion")
    st.write("Upload technical docs to initialize the Agent's memory.")
    uploaded_file = st.file_uploader("Upload Phison/MaiStorage PDF", type="pdf")
    
    if uploaded_file:
        # Check if the database is already built in this session to avoid re-indexing
        if "db" not in st.session_state:
            with st.spinner("Agent is reading and indexing..."):
                # Save temporary file locally
                with open("temp_demo.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load PDF
                loader = PyPDFLoader("temp_demo.pdf")
                data = loader.load()
                
                # CHUNKING: Using 2000 size to minimize API calls (Rate Limit Protection)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = text_splitter.split_documents(data)
                
                # EMBEDDING & STORAGE
                st.session_state.db = Chroma.from_documents(
                    documents=chunks, 
                    embedding=embeddings,
                    collection_name="maistorage_demo"
                )
                st.success(f"✅ Knowledge Base Ready: {len(chunks)} chunks.")

# --- 3. CHAT INTERFACE & RERANKING LOGIC ---
st.title("🤖 MaiStorage Technical Agent")
st.caption("Simplified RAG Prototype for Internal Documentation")

query = st.chat_input("Ask a technical question...")

if query:
    with st.chat_message("user"):
        st.write(query)
    
    if "db" not in st.session_state:
        st.warning("Please upload a PDF spec sheet in the sidebar to begin.")
    else:
        with st.chat_message("assistant"):
            with st.status("Agent Reasoning..."):
                # STEP 1: RETRIEVAL (Fetch 5 candidates)
                initial_docs = st.session_state.db.similarity_search(query, k=5)
                
                # STEP 2: SIMPLE RERANKING (Keyword-based)
                query_words = query.lower().split()
                reranked_docs = sorted(
                    initial_docs, 
                    key=lambda d: sum(word in d.page_content.lower() for word in query_words), 
                    reverse=True
                )
                
                # STEP 3: GENERATION (Top 3 chunks)
                context = "\n\n".join([d.page_content for d in reranked_docs[:3]])
                
                prompt = f"""Answer based ONLY on the context. If not found, say you don't know.
                
                CONTEXT:
                {context}
                
                QUESTION: {query}"""
                
                response = llm.invoke(prompt)
            
            st.markdown(response.content)
