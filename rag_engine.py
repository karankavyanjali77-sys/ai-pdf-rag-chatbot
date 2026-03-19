from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain_groq import ChatGroq


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
    # LLM (Groq)
    llm = ChatGroq(
        groq_api_key="YOUR_GROQ_API_KEY",
        model_name="llama3-70b-8192"
    )

    # Prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an AI assistant. Answer ONLY from the context.

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
