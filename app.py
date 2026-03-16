import streamlit as st
import tempfile
from rag_engine import build_vectorstore, build_chatbot

st.set_page_config(
    page_title="AI PDF Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("📚 AI Multi-PDF Knowledge Assistant")

st.markdown(
"""
Upload one or more PDFs and ask questions about them.

This chatbot uses **Retrieval Augmented Generation (RAG)** to answer questions from your documents.
"""
)

# session states
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# sidebar
with st.sidebar:

    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Process PDFs"):

        if uploaded_files:

            pdf_paths = []

            for file in uploaded_files:
                temp = tempfile.NamedTemporaryFile(delete=False)
                temp.write(file.read())
                pdf_paths.append(temp.name)

            with st.spinner("Creating embeddings..."):

                vectorstore = build_vectorstore(pdf_paths)
                qa_chain = build_chatbot(vectorstore)

                st.session_state.vectorstore = vectorstore
                st.session_state.qa_chain = qa_chain

            st.success("Documents processed successfully!")

        else:
            st.warning("Please upload at least one PDF.")

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# display chat history
for role, message in st.session_state.chat_history:

    if role == "user":
        with st.chat_message("user"):
            st.write(message)

    else:
        with st.chat_message("assistant"):
            st.write(message)

# chat input
question = st.chat_input("Ask something about your documents...")

if question and st.session_state.qa_chain:

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            result = st.session_state.qa_chain({"question": question})

            answer = result["answer"]

            st.write(answer)

            # show sources
            if "source_documents" in result:

                with st.expander("Sources"):

                    for doc in result["source_documents"]:
                        st.write(
                            f"Page {doc.metadata.get('page', 'Unknown')}"
                        )

    st.session_state.chat_history.append(("user", question))
    st.session_state.chat_history.append(("ai", answer))

elif question and not st.session_state.qa_chain:
    st.warning("Please upload and process PDFs first.")
