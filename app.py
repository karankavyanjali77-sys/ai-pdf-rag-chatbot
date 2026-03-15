import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

st.set_page_config(page_title="AI PDF Chatbot", layout="wide")
st.title("📄 AI PDF Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "all_text" not in st.session_state:
    st.session_state.all_text = ""

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files and not st.session_state.pdf_ready:
    all_documents = []
    all_text_parts = []

    with st.spinner("Processing PDFs..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            for d in docs:
                if d.page_content.strip():
                    all_text_parts.append(d.page_content)

            all_documents.extend(docs)
            os.remove(tmp_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        split_docs = splitter.split_documents(all_documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = FAISS.from_documents(split_docs, embeddings)
        st.session_state.retriever = db.as_retriever(search_kwargs={"k": 6})
        st.session_state.all_text = "\n\n".join(all_text_parts)
        st.session_state.pdf_ready = True

    st.success("PDFs processed successfully.")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key="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"
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask something about the PDF")

if question and st.session_state.pdf_ready:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    q_lower = question.lower()

    summary_keywords = [
        "summarize",
        "summary",
        "what is this pdf about",
        "what is in this pdf",
        "overview of this pdf",
        "explain this pdf"
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            if any(keyword in q_lower for keyword in summary_keywords):
                context = st.session_state.all_text[:12000]

                prompt = f"""
You are an AI assistant.
The following text is extracted from a PDF.

Your task:
1. Summarize the PDF clearly.
2. Mention the main topics.
3. Give a short structured overview.
4. Do not say "Not found in document" unless the text is empty.

PDF Text:
{context}

User request:
{question}
"""
                response = llm.invoke(prompt)
                answer = response.content
                st.write(answer)

            else:
                retrieved_docs = st.session_state.retriever.invoke(question)

                context = ""
                sources = []

                for doc in retrieved_docs:
                    context += doc.page_content + "\n\n"
                    sources.append(doc.metadata.get("page", "unknown"))

                prompt = f"""
You are an AI assistant answering questions from a PDF.

Rules:
- Answer only from the provided context.
- If the answer is clearly present, answer naturally and clearly.
- If the answer is partially present, say what is available.
- Only say "Not found in document" if the context truly does not contain the answer.

Context:
{context}

Question:
{question}
"""

                response = llm.invoke(prompt)
                answer = response.content
                st.write(answer)
                st.info(f"Source pages: {sources}")

    st.session_state.messages.append({"role": "assistant", "content": answer})

elif question and not st.session_state.pdf_ready:
    st.warning("Please upload and process at least one PDF first.")
