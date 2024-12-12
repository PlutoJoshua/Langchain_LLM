import os
import sys

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings

# .env 파일 로드
env_path = ".../.env"
load_dotenv(env_path)

def initialize_vectorstore():
    index_name = os.getenv("PINECONE_INDEX")
    embeddings = OpenAIEmbeddings()
    return Pinecone.from_existing_index(index_name, embeddings)

if __name__ == "__main__":
    file_path = r"C:\Users\JOSHUA\Documents\Langchain_LLM\data\Lecture 01. Introduction to NLP.pdf"
    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()
    print("Loaded", len(raw_docs))
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs = text_splitter.split_documents(raw_docs)
    print("split ", len(docs))

    vectorstore = initialize_vectorstore()
    vectorstore.add_documents(docs)