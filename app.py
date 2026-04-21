import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# --- 1. CONFIGURATION ---
load_dotenv() # Loads from your local .env file

# Priority: Streamlit Cloud Secrets -> Local .env -> None
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Missing API Key! Please add it to your .env file or Streamlit Secrets.")
    st.stop()

# Initialize AI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# --- 2. AGENT STATE ---
class AgentState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# --- 3. AGENT NODES ---
def retrieve(state):
    """Fetch documents from the knowledge base."""
    # Note: Ensure chroma_db exists by running ingest_data.py first
    if not os.path.exists("./chroma_db"):
        st.warning("Knowledge base not found. Please run ingest_data.py first!")
        return {"documents": []}
        
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    docs = db.similarity_search(state["question"], k=3)
    return {"documents": docs}

def grade_and_generate(state):
    """Reasoning node: Checks if we have data, then answers."""
    if not state["documents"]:
        return {"generation": "I'm sorry, I don't have internal data on that topic."}
    
    context = "\n\n".join([d.page_content for d in state["documents"]])
    prompt = f"Answer based ONLY on this context: {context}\n\nQuestion: {state['question']}"
    
    response = llm.invoke(prompt)
    return {"generation": response.content}

# --- 4. THE GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", grade_and_generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
agent = workflow.compile()

# --- 5. STREAMLIT UI ---
st.title("🤖 MaiStorage Agentic RAG")
query = st.chat_input("Ask a technical question...")

if query:
    with st.chat_message("user"): st.write(query)
    with st.chat_message("assistant"):
        with st.status("Thinking..."):
            result = agent.invoke({"question": query})
        st.write(result["generation"])
