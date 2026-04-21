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

# Display historical messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there are references saved for this specific bot message, show them in an expander
        if "references" in message:
            with st.expander("📚 View References"):
                st.markdown(message["references"])

# --- 5. CHAT LOGIC ---
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
                docs = st.session_state.db.similarity_search(query, k=3)
                
                # CLEANER CITATION LOGIC: Group by page to avoid redundancy
                pages_found = {}
                for doc in docs:
                    p = doc.metadata.get("page", 0) + 1
                    if p not in pages_found:
                        pages_found[p] = []
                    pages_found[p].append(doc.page_content[:300].replace("\n", " "))

                # Format the "Hidden" reference string
                ref_string = ""
                for page, snippets in sorted(pages_found.items()):
                    ref_string += f"**Page {page}**\n"
                    for s in snippets:
                        ref_string += f"> ...{s}...\n\n"
                
                context = "\n\n".join([f"[Source Page {d.metadata.get('page')+1}] {d.page_content}" for d in docs])
                history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-5:-1]])
                
                prompt = f"""Use ONLY the CONTEXT to answer the question.
                CHAT HISTORY: {history_str}
                CONTEXT: {context}
                QUESTION: {query}"""
                
                response = llm.invoke(prompt)
                answer = response.content

            # Display the answer
            st.markdown(answer)
            
            # Display the "Small Tab" for references
            with st.expander("📚 View References"):
                st.markdown(ref_string)
            
            # Save to history with the references attached
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": answer, 
                "references": ref_string
            })
