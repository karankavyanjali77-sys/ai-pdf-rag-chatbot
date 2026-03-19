import streamlit as st
import os
from rag_engine import build_vectorstore, build_chatbot

st.set_page_config(page_title="AI PDF Chatbot", layout="wide")

st.title("📄 AI PDF Chatbot (LangChain + Groq)")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Save temp file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing PDF..."):
        vectorstore = build_vectorstore("temp.pdf")
        qa_chain = build_chatbot(vectorstore)

    st.success("PDF processed successfully!")

    # Chat input
    query = st.text_input("Ask a question from the PDF")

    if query:
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({"query": query})

        st.subheader("Answer:")
        st.write(response["result"])
