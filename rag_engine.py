import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Use langchain_huggingface if available, fallback to community
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


def build_vectorstore(pdf_path):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    split_docs = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Vector DB
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


def build_chatbot(vectorstore):
    # LLM (Groq) — reads API key from environment variable
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please set it in your .env file or environment variables."
        )

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-70b-8192"
    )

    # Prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an AI assistant. Answer ONLY from the context below.
If the answer is not found in the context, say "I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain