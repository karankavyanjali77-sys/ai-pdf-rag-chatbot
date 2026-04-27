import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import build_vectorstore, build_chatbot

load_dotenv()

st.set_page_config(page_title="AI PDF Chatbot", layout="wide")
st.title("📄 AI PDF Chatbot (LangChain + Groq)")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    if st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.qa_chain = None
        st.session_state.last_uploaded_file = uploaded_file.name

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    if st.session_state.qa_chain is None:
        with st.spinner("Processing PDF..."):
            try:
                vectorstore = build_vectorstore("temp.pdf")
                st.session_state.qa_chain = build_chatbot(vectorstore)
                st.success("✅ PDF processed successfully!")
            except ValueError as e:
                st.error(f"❌ Configuration error: {e}")
                st.stop()
            except Exception as e:
                st.error(f"❌ Error processing PDF: {e}")
                st.stop()

if st.session_state.qa_chain:
    st.subheader("💬 Chat with your PDF")
    query = st.text_input("Ask a question:")

    if query:
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.qa_chain.invoke({"query": query})
                st.markdown("### 🧠 Answer:")
                st.write(response["result"])
            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")