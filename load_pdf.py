from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/book.pdf")

documents = loader.load()

print(len(documents))
print(documents[0].page_content)