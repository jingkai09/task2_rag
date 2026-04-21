import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. CONFIGURATION ---
# Use your new, private API Key
API_KEY = "AIzaSyBpaYk9WSnjA8puTdlL-qSIglBX3Kj82NE" 
os.environ["GOOGLE_API_KEY"] = API_KEY

st.set_page_config(page_title="MaiStorage Multi-Doc Agent", layout="wide")

# Initialize Gemini 2.5 Flash & Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY, temperature=0.1)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=API_KEY)

# --- 2. SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = [] # Tracks names of files already in the DB

# --- 3. SIDEBAR: MULTI-DOC INGESTION ---
with st.sidebar:
    st.header("📂 Knowledge Management")
    st.write("Upload multiple PDFs to build a cross-document knowledge base.")
    uploaded_file = st.file_uploader("Upload Phison/MaiStorage PDF", type="pdf")
    
    if uploaded_file:
        # Only process if this specific file hasn't been added yet
        if uploaded_file.name not in st.session_state.uploaded_files:
            with st.spinner(f"Reading {uploaded_file.name}..."):
                # Unique temp path for each file to prevent overwriting
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                loader = PyPDFLoader(temp_path)
                data = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
                chunks = text_splitter.split_documents(data)
                
                # If first doc, create the DB. If subsequent, add to it.
                if "db" not in st.session_state:
                    st.session_state.db = Chroma.from_documents(
                        documents=chunks, 
                        embedding=embeddings, 
                        collection_name="demo_collection"
                    )
                else:
                    st.session_state.db.add_documents(chunks)
                
                st.session_state.uploaded_files.append(uploaded_file.name)
                st.success(f"✅ Added {uploaded_file.name}")
    
    # Display list of current knowledge base
    if st.session_state.uploaded_files:
        st.write("**Current Knowledge Base:**")
        for f_name in st.session_state.uploaded_files:
            st.caption(f"• {f_name}")

    if st.button("🗑️ Reset All Data"):
        st.session_state.chat_history = []
        st.session_state.uploaded_files = []
        if "db" in st.session_state:
            del st.session_state.db
        st.rerun()

# --- 4. UI DISPLAY ---
st.title("🤖 MaiStorage Technical Agent")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "references" in message:
            with st.expander("📚 View References"):
                st.markdown(message["references"])

# --- 5. CONVERSATIONAL LOGIC ---
query = st.chat_input("Ask a question across all uploaded documents...")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    if "db" not in st.session_state:
        st.warning("Please upload at least one PDF first.")
    else:
        with st.chat_message("assistant"):
            with st.status("Agentic Cross-Document Retrieval..."):
                # Retrieval across all chunks in the collection
                docs = st.session_state.db.similarity_search(query, k=4)
                
                # Group references by page and source file
                sources_found = {}
                for doc in docs:
                    src = os.path.basename(doc.metadata.get("source", "Unknown"))
                    pg = doc.metadata.get("page", 0) + 1
                    key = f"{src} (Page {pg})"
                    if key not in sources_found:
                        sources_found[key] = []
                    sources_found[key].append(doc.page_content[:250].replace("\n", " "))

                ref_string = ""
                for source, snippets in sources_found.items():
                    ref_string += f"**{source}**\n"
                    for s in snippets:
                        ref_string += f"> ...{s}...\n\n"
                
                context = "\n\n".join([f"[Source: {d.metadata.get('source')} Page {d.metadata.get('page')+1}] {d.page_content}" for d in docs])
                history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-5:-1]])
                
                prompt = f"""You are a technical expert. Use the CONTEXT to answer the question.
                If the question spans multiple documents, synthesize the information clearly.
                
                CHAT HISTORY: {history_str}
                CONTEXT: {context}
                QUESTION: {query}"""
                
                response = llm.invoke(prompt)
                answer = response.content

            st.markdown(answer)
            with st.expander("📚 View References"):
                st.markdown(ref_string)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": answer, 
                "references": ref_string
            })
