import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. CONFIGURATION ---
API_KEY = "AIzaSyBpaYk9WSnjA8puTdlL-qSIglBX3Kj82NE" 
os.environ["GOOGLE_API_KEY"] = API_KEY

st.set_page_config(page_title="MaiStorage Agentic RAG", layout="wide")

# Initialize Gemini 2.5 Flash & Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY, temperature=0.1)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=API_KEY)

# --- 2. SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 

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
            
            # Optimized splitting for better citation accuracy
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
            chunks = text_splitter.split_documents(data)
            
            st.session_state.db = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings, 
                collection_name="demo_collection"
            )
            st.success("✅ Knowledge Base Ready")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# --- 4. UI DISPLAY ---
st.title("🤖 MaiStorage Technical Agent")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. CHAT LOGIC WITH CITATIONS ---
query = st.chat_input("Ask a technical question...")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    if "db" not in st.session_state:
        st.warning("Please upload a PDF first.")
    else:
        with st.chat_message("assistant"):
            with st.status("Agentic Retrieval & Reranking..."):
                # STEP 1: RETRIEVAL
                docs = st.session_state.db.similarity_search(query, k=3)
                
                # STEP 2: EXTRACT CITATIONS FROM METADATA
                # PyPDFLoader automatically provides 'page' and 'source' in metadata
                references = []
                for doc in docs:
                    page_num = doc.metadata.get("page", "Unknown") + 1 # Convert 0-index to 1-index
                    ref_text = f"📄 Page {page_num}: \"{doc.page_content[:150]}...\""
                    references.append(ref_text)
                
                context = "\n\n".join([f"[Source Page {d.metadata.get('page')+1}] {d.page_content}" for d in docs])
                
                # STEP 3: CONSTRUCT HISTORY
                history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-5:-1]])
                
                # STEP 4: GENERATION
                prompt = f"""You are a technical assistant. Use ONLY the CONTEXT to answer the question.
                Mention specific page numbers in your answer if relevant.
                
                CHAT HISTORY: {history_str}
                CONTEXT FROM PDF: {context}
                QUESTION: {query}"""
                
                response = llm.invoke(prompt)
                
                # Format final output with a "Sources" footer
                final_answer = response.content
                reference_section = "\n\n---\n**Sources used for this answer:**\n" + "\n".join([f"* {r}" for r in references])
                full_display = final_answer + reference_section

            st.markdown(full_display)
            st.session_state.chat_history.append({"role": "assistant", "content": full_display})
