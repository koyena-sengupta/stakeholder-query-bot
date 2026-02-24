import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from config import *

def load_documents():
    documents = []
    
    for file in os.listdir(DATA_PATH):
        path = os.path.join(DATA_PATH, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".txt"):
            loader = TextLoader(path)
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue

        documents.extend(loader.load())

    return documents


def ingest():
    print("🔹 Loading documents...")
    documents = load_documents()

    print("🔹 Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    print("🔹 Creating embeddings...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    print("🔹 Storing in vector DB...")
    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    print("✅ Ingestion complete!")


if __name__ == "__main__":
    ingest()
