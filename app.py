import streamlit as st
import os
from rag_engine import build_vectorstore, build_chatbot

st.set_page_config(page_title="AI PDF Chatbot", layout="wide")

st.title("📄 AI PDF Chatbot (LangChain + Groq)")

# Session state (IMPORTANT)
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Save temp file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Build only once
    if st.session_state.qa_chain is None:
        with st.spinner("Processing PDF..."):
            vectorstore = build_vectorstore("temp.pdf")
            st.session_state.qa_chain = build_chatbot(vectorstore)

        st.success("✅ PDF processed successfully!")

# Chat section
if st.session_state.qa_chain:
    st.subheader("💬 Chat with your PDF")

    query = st.text_input("Ask a question:")

    if query:
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.invoke({
                "question": query  
            })

        st.markdown("### 🧠 Answer:")
        st.write(response["result"])
