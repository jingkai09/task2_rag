import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. CONFIGURATION ---
# Replace this with your NEW API Key from Google AI Studio
API_KEY = "AIzaSyALkdgFSrUciFaE--xZdDhQ0OfJsji_Opg" 

st.set_page_config(page_title="MaiStorage AI Assistant", layout="wide")

# Initialize Gemini 2.5 Flash (Faster & Higher Intelligence for Reranking)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=API_KEY,
    temperature=0.1
)

# Initialize Embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2.5", 
    google_api_key=API_KEY
)

# --- 2. SIDEBAR: DOCUMENT INGESTION ---
with st.sidebar:
    st.header("📂 Data Ingestion")
    st.write("Upload technical docs to initialize the Agent's memory.")
    uploaded_file = st.file_uploader("Upload Phison/MaiStorage PDF", type="pdf")
    
    if uploaded_file:
        with st.spinner("Agent is reading and indexing..."):
            # Save temporary file locally
            with open("temp_demo.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load PDF
            loader = PyPDFLoader("temp_demo.pdf")
            data = loader.load()
            
            # CHUNKING: Using 2000 size to minimize API calls (Rate Limit Protection)
            # 
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            chunks = text_splitter.split_documents(data)
            
            # EMBEDDING & STORAGE
            # Creating a fresh in-memory collection for the demo
            st.session_state.db = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings,
                collection_name="maistorage_demo"
            )
            st.success(f"✅ Knowledge Base Ready: {len(chunks)} chunks.")

# --- 3. CHAT INTERFACE & RERANKING LOGIC ---
st.title("🤖 MaiStorage Technical Agent")
st.caption("Powered by Gemini 2.5 Flash & Agentic RAG")

query = st.chat_input("Ask a technical question about aiDAPTIV+ or NAND specs...")

if query:
    with st.chat_message("user"):
        st.write(query)
    
    if "db" not in st.session_state:
        st.warning("Please upload a PDF spec sheet in the sidebar to begin.")
    else:
        with st.chat_message("assistant"):
            with st.status("Agent Reasoning (Retrieve -> Rerank -> Generate)..."):
                # STEP 1: RETRIEVAL (Fetch 5 most mathematically similar chunks)
                initial_docs = st.session_state.db.similarity_search(query, k=5)
                
                # STEP 2: SIMPLE AGENTIC RERANKING
                # We prioritize chunks that contain the actual words from your query
                query_words = query.lower().split()
                reranked_docs = sorted(
                    initial_docs, 
                    key=lambda d: sum(word in d.page_content.lower() for word in query_words), 
                    reverse=True
                )
                
                # STEP 3: GENERATION (Use top 3 reranked chunks for context)
                context = "\n\n".join([d.page_content for d in reranked_docs[:3]])
                
                prompt = f"""You are a MaiStorage Technical Expert. 
                Answer the question based ONLY on the provided context.
                If you cannot find the answer, politely state that it's not in the document.
                
                CONTEXT:
                {context}
                
                QUESTION: {query}"""
                
                response = llm.invoke(prompt)
            
            # Display final answer
            st.markdown(response.content)
            
            # Optional: Show which chunks were used
            with st.expander("View Reranked Sources"):
                for i, doc in enumerate(reranked_docs[:3]):
                    st.write(f"**Source {i+1}:** {doc.page_content[:300]}...")
