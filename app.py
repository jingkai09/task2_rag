import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. CONFIGURATION (DEMO MODE) ---
# 1. Paste your NEW key here
API_KEY = "AIzaSyA0SmLYBntwkTSGwY68PIsuouaodQ5gDFM" 

# 2. Force the environment to recognize it (Fixes 'Invalid Key' errors)
os.environ["GOOGLE_API_KEY"] = API_KEY

st.set_page_config(page_title="MaiStorage AI Assistant", layout="wide")

# Initialize Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=API_KEY,
    temperature=0.1
)

# Initialize stable Embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    google_api_key=API_KEY
)

# --- 2. SIDEBAR: DATA INGESTION ---
with st.sidebar:
    st.header("📂 Knowledge Management")
    uploaded_file = st.file_uploader("Upload Phison/MaiStorage PDF", type="pdf")
    
    if uploaded_file:
        if "db" not in st.session_state:
            with st.spinner("Processing technical document..."):
                # Save temporary file locally
                with open("temp_demo.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                loader = PyPDFLoader("temp_demo.pdf")
                data = loader.load()
                
                # CHUNKING: Using 2000 size to minimize API calls (Rate Limit Protection)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = text_splitter.split_documents(data)
                
                # EMBEDDING & STORAGE (In-memory for the demo)
                st.session_state.db = Chroma.from_documents(
                    documents=chunks, 
                    embedding=embeddings,
                    collection_name="demo_collection"
                )
                st.success(f"✅ Knowledge Base Ready")

# --- 3. CHAT INTERFACE & RERANKING ---
st.title("🤖 MaiStorage Technical Agent")

query = st.chat_input("Ask a technical question about the uploaded document...")

if query:
    with st.chat_message("user"):
        st.write(query)
    
    if "db" not in st.session_state:
        st.warning("Please upload a PDF in the sidebar first.")
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
                
                # STEP 3: GENERATION (Top 2 chunks to avoid context bloat)
                context = "\n\n".join([d.page_content for d in reranked_docs[:2]])
                
                prompt = f"""Use ONLY the following context to answer the question.
                If the answer is not in the context, say you don't know.
                
                CONTEXT:
                {context}
                
                QUESTION: {query}"""
                
                response = llm.invoke(prompt)
            
            st.markdown(response.content)
