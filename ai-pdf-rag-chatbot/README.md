# AI PDF Chatbot using RAG

An AI-powered PDF chatbot that lets users upload one or more PDF files, ask questions in natural language, and get answers based only on the document content.

This project uses a Retrieval-Augmented Generation (RAG) pipeline with LangChain, FAISS, Sentence Transformers, Groq, and Streamlit.

## Features

- Upload one or multiple PDF files
- Ask questions about the uploaded documents
- Get document-based answers using RAG
- Summarize the PDF contents
- Show source pages for retrieved answers
- Clean chat-style interface using Streamlit

## Tech Stack

- Streamlit
- LangChain
- FAISS
- Sentence Transformers
- Groq LLM
- PyPDF

## Project Structure

```bash
.
├── app.py
├── requirements.txt
└── README.md
