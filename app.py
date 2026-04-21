import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Literal

# 1. Setup Environment
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 2. Graph State
class AgentState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    iteration: int

# 3. Nodes (The "Brain" of the Agent)
def retrieve(state: AgentState):
    """Retrieve documents from the vector store."""
    st.write("🔍 **Agent:** Accessing internal knowledge base...")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    docs = vectorstore.similarity_search(state["question"], k=3)
    return {"documents": docs, "iteration": state.get("iteration", 0) + 1}

def grade_relevance(state: AgentState) -> Literal["generate", "transform_query"]:
    """Grades whether the retrieved docs are useful."""
    st.write("⚖️ **Agent:** Evaluating document relevance...")
    
    # In a real Agentic RAG, we'd use an LLM to score this (0.85 threshold).
    # For the demo, we check if any docs were found.
    if not state["documents"] or state["iteration"] > 2:
        return "generate" # Proceed to answer or admit failure
    
    # Example logic: if the query is too vague, we transform it.
    return "generate"

def generate(state: AgentState):
    """Generate the final grounded answer with citations."""
    st.write("✍️ **Agent:** Finalizing response based on facts...")
    context = "\n\n".join([f"[Source: {d.metadata.get('source', 'Unknown')}] {d.page_content}" for d in state["documents"]])
    
    prompt = f"""You are a MaiStorage Technical Assistant. 
    Using the context below, answer the question. 
    Strictly avoid hallucinations. If the info isn't there, say so.
    
    CONTEXT: {context}
    QUESTION: {state['question']}"""
    
    response = llm.invoke(prompt)
    return {"generation": response.content}

# 4. Building the Graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_conditional_edges("retrieve", grade_relevance, {
    "generate": "generate",
    "transform_query": "retrieve" # This creates the loop
})
workflow.add_edge("generate", END)
agent_app = workflow.compile()

# 5. Streamlit Interface
st.title("MaiStorage Agentic RAG Prototype")
st.info("Self-correcting retrieval system for technical documentation.")

user_input = st.text_input("Enter your technical query:")
if user_input:
    with st.status("Agentic Workflow Active...", expanded=True) as status:
        final_state = agent_app.invoke({"question": user_input, "iteration": 0})
        status.update(label="Workflow Complete", state="complete")
    
    st.subheader("Response")
    st.markdown(final_state["generation"])
