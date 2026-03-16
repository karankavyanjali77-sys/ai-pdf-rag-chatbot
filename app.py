import streamlit as st
import tempfile
from rag_engine import build_vectorstore, build_chatbot

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="🤖"
)

st.title("📚 AI Multi-PDF Knowledge Assistant")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    pdf_paths = []

    for file in uploaded_files:

        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(file.read())
        pdf_paths.append(temp.name)

    vectorstore = build_vectorstore(pdf_paths)

    qa_chain = build_chatbot(vectorstore)

    question = st.text_input("Ask a question about your documents")

    if question:

        result = qa_chain({"question": question})

        st.write("### Answer")
        st.write(result["answer"])