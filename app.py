import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. CONFIGURATION ---
API_KEY = "AIzaSyA0SmLYBntwkTSGwY68PIsuouaodQ5gDFM" 
os.environ["GOOGLE_API_KEY"] = API_KEY

st.set_page_config(page_title="MaiStorage Conversational Agent", layout="wide")

# Initialize Gemini 2.5 Flash & Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY, temperature=0.1)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=API_KEY)

# --- 2. SESSION STATE INITIALIZATION ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # Stores all messages for the UI and LLM context

# --- 3. SIDEBAR: DATA INGESTION ---
with st.sidebar:
    st.header("📂 Knowledge Management")
    uploaded_file = st.file_uploader("Upload Phison/MaiStorage PDF", type="pdf")
    
    if uploaded_file and "db" not in st.session_state:
        with st.spinner("Processing technical document..."):
            with open("temp_demo.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader("temp_demo.pdf")
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            chunks = text_splitter.split_documents(data)
            st.session_state.db = Chroma.from_documents(
                documents=chunks, embedding=embeddings, collection_name="demo_collection"
            )
            st.success("✅ Knowledge Base Ready")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# --- 4. CHAT DISPLAY ---
st.title("🤖 MaiStorage Conversational Agent")

# Display every message in the history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. CHAT LOGIC ---
query = st.chat_input("Ask a technical question...")

if query:
    # 1. Show user message immediately
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    if "db" not in st.session_state:
        st.warning("Please upload a PDF first.")
    else:
        with st.chat_message("assistant"):
            with st.status("Thinking..."):
                # STEP 1: RETRIEVAL
                docs = st.session_state.db.similarity_search(query, k=3)
                context = "\n\n".join([d.page_content for d in docs])
                
                # STEP 2: CONSTRUCT HISTORY FOR LLM
                # We format the last 4 messages to keep the context relevant
                history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-5:-1]])
                
                # STEP 3: CONTEXTUAL GENERATION
                prompt = f"""You are a technical assistant. Use the CONTEXT and the CHAT HISTORY to answer the question.
                
                CHAT HISTORY:
                {history_str}
                
                NEW CONTEXT FROM PDF:
                {context}
                
                USER QUESTION: {query}"""
                
                response = llm.invoke(prompt)
                full_response = response.content
            
            # 2. Show and save assistant response
            st.markdown(full_response)
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
