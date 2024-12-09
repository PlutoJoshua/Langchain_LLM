from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY")
print("API Key:", api_key)

# 레시피 데이터 모델 정의
class Recipe(BaseModel):
    ingredients: list[str] = Field(description="Ingredients of the dish")
    steps: list[str] = Field(description="Steps to make the dish")

# Pydantic Output Parser 설정
output_parser = PydanticOutputParser(pydantic_object=Recipe)

# 프롬프트 템플릿 정의
template = """다음 요리의 레시피를 생각해 주세요.

{format_instructions}

요리: {dish}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["format_instructions", "dish"]
)

# ChatOpenAI 모델 설정
chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)

# LLMChain 설정
chain = LLMChain(
    prompt=prompt,
    llm=chat,
    output_parser=output_parser
)

# 입력 데이터 생성
inputs = {
    "dish": "순두부찌개",
    "format_instructions": output_parser.get_format_instructions()
}

# 체인 실행
recipe = chain.invoke(inputs)

# 결과 출력
print(type(recipe))
print(recipe)