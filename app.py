import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# 1. Setup & Config
load_dotenv()
st.set_page_config(page_title="MaiStorage Agentic RAG", layout="wide")

# Initialize Gemini Models
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# State Definition for LangGraph
class AgentState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    is_relevant: bool

# 2. Sidebar: Document Ingestion
with st.sidebar:
    st.header("🏢 MaiStorage Knowledge Base")
    uploaded_file = st.file_uploader("Upload Internal Tech Specs (PDF)", type="pdf")
    
    if uploaded_file:
        with st.spinner("Indexing document into ChromaDB..."):
            # Save temp file for loader
            with open("temp_knowledge.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader("temp_knowledge.pdf")
            data = loader.load()
            
            # Recursive Splitting to maintain technical context
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)
            
            # Persist in local ChromaDB
            vectorstore = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings, 
                persist_directory="./chroma_db"
            )
            st.success("Successfully indexed for retrieval.")

# 3. Agentic Nodes (The Reasoning Logic)
def retrieve_node(state):
    st.write("🔍 **Agent:** Searching vector database...")
    v_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    docs = v_db.similarity_search(state["question"], k=3)
    return {"documents": docs}

def grade_node(state):
    st.write("⚖️ **Agent:** Grading relevance of retrieved facts...")
    # Logic: If no docs found, we flag for 'no-answer' or 'rewrite'
    is_rel = len(state["documents"]) > 0
    return {"is_relevant": is_rel}

def generate_node(state):
    st.write("✍️ **Agent:** Formulating grounded response...")
    # Join documents and include source metadata for citations
    context = "\n\n".join([f"[Source: {d.metadata.get('source', 'Doc')}] {d.page_content}" for d in state["documents"]])
    
    prompt = f"""You are a MaiStorage Technical Assistant. 
    Answer strictly using the context below. Include citations in brackets.
    If the context doesn't have the answer, admit you don't know based on internal data.
    
    CONTEXT: {context}
    QUESTION: {state['question']}"""
    
    response = llm.invoke(prompt)
    return {"generation": response.content}

# 4. Building the Graph Loop
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges(
    "grade",
    lambda x: "generate" if x["is_relevant"] else END
)
workflow.add_edge("generate", END)
agent_executor = workflow.compile()

# 5. Chat UI
st.title("🤖 Agentic RAG Prototype")
st.markdown("---")

query = st.chat_input("Ask about MaiStorage technology...")

if query:
    with st.chat_message("user"):
        st.write(query)
        
    with st.chat_message("assistant"):
        with st.status("Agent thinking...", expanded=True) as status:
            result = agent_executor.invoke({"question": query})
            status.update(label="Reasoning Complete", state="complete")
        
        # Display Final Answer
        if "generation" in result:
            st.write(result["generation"])
        else:
            st.error("No relevant information found in the internal knowledge base.")
