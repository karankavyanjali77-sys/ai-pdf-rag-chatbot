import streamlit as st
import tempfile
from rag_engine import build_vectorstore, build_chatbot

st.set_page_config(
    page_title="AI PDF Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("📚 AI Multi-PDF Knowledge Assistant")

# store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader(
    "Upload your PDFs",
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

    question = st.chat_input("Ask a question about your documents")

    if question:

        result = qa_chain({"question": question})

        answer = result["answer"]

        st.session_state.chat_history.append(("user", question))
        st.session_state.chat_history.append(("ai", answer))

# display chat history
for role, message in st.session_state.chat_history:

    if role == "user":
        with st.chat_message("user"):
            st.write(message)

    else:
        with st.chat_message("assistant"):
            st.write(message)
