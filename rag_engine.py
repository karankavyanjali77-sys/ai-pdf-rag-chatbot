from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv

load_dotenv()


def build_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    split_docs = splitter.split_documents(docs)

    embeddings = FakeEmbeddings(size=384)

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    return vectorstore


def build_chatbot(vectorstore):
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192"   # ✅ better choice
    )

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

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain
