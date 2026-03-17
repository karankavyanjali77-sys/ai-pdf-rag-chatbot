from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM


def build_vectorstore(pdf_paths):

    docs = []

    for pdf in pdf_paths:
        loader = PyPDFLoader(pdf)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(splits, embeddings)

    return vectorstore


def build_chatbot(vectorstore):

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    llm = OllamaLLM(
        model="mistral",   # or "llama3"
        temperature=0.5
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

    return qa
