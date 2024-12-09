from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# .env 파일 로드
env_path = "../.env"
load_dotenv(env_path)

# OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI 임베딩과 LLM 설정
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# PDF 로드 및 텍스트 분할
def load_and_split_pdf(pdf_folder):
    documents = []
    text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=200)

    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file_name)
            loader = PyPDFLoader(file_path)
            docs = loader.load_and_split(text_splitter=text_splitter)
            documents.extend(docs)

    return documents

# PDF 텍스트 임베딩 및 FAISS 저장
def embed_and_store_in_faiss(documents, index_path):
    # FAISS 벡터스토어 생성
    vectorstore = FAISS.from_documents(documents, embeddings)
    # 벡터스토어 저장
    vectorstore.save_local(index_path)
    print(f"FAISS index saved to {index_path}")
    return vectorstore

# 쿼리 수행 후 결과를 GPT-4o-mini 모델로 처리
def query_with_gpt(query, pdf_folder, index_path, k):
    # PDF에서 텍스트 로드 및 임베딩 생성
    documents = load_and_split_pdf(pdf_folder)
    
    # FAISS 벡터스토어 저장 및 로드
    vectorstore = embed_and_store_in_faiss(documents, index_path)
    
    # FAISS를 사용하여 쿼리 관련 문서 검색 (k개 설정)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    relevant_docs = retriever.invoke(query)
    
    print(f"Found {len(relevant_docs)} relevant documents.")
    
    # 검색된 관련 문서들을 프롬프트로 작성
    prompt = "다음 문서들을 바탕으로 질문에 답해주세요:\n"
    for i, doc in enumerate(relevant_docs):
        prompt += f"{i+1}. {doc.page_content}\n"

    prompt += f"질문: {query}"

    # LLM에 넘겨서 답변 받기
    response = llm.invoke(prompt)
    return response

# 사용 예시
pdf_folder = "./data"  # PDF 파일이 있는 폴더
index_path = "./faiss_index"  # FAISS 인덱스 저장 경로

# 쿼리 실행
query = "What is NLP?"
response = query_with_gpt(query, pdf_folder, index_path, 5)  # k=5로 설정
print(f"GPT-4o-mini 답변: {response}")