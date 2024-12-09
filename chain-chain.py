from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# .env 파일 로드
env_path = "../.env"
load_dotenv(env_path)

# OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY")

# ChatOpenAI 객체 생성
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# COT (Chain of Thought) 프롬프트 템플릿 정의
cot_template = """다음 질문에 답하세요.

질문: {question}

단계별로 생각해 봅시다.
"""

# 프롬프트 템플릿에 필요한 변수와 템플릿을 설정
cot_prompt = PromptTemplate(
    input_variables=["question"],
    template=cot_template,
)

# LLMChain 객체 생성 (COT 프롬프트와 ChatOpenAI 연결)
cot_chain = LLMChain(llm=chat, prompt=cot_prompt)

# COT 체인 실행
cot_chain_result = cot_chain.invoke(
    "나는 시장에서 사과 10개를 샀다. 누나에게 2개, 엄마에게 2개 주었다. 그 다음 사과를 4개 더 사서 2개를 먹었다. 남은 갯수는 ?"
)
# COT 체인 결과 출력
print(f'- cot: {cot_chain_result['text']}\n')

# 요약 프롬프트 템플릿 정의
summarize_template = """다음 문장을 간단히 요약하세요.

{input}

"""

# LLMChain 객체 생성 (요약 프롬프트와 ChatOpenAI 연결)
summarize_prompt = PromptTemplate(
    input_variables=["input"],
    template=summarize_template
)

# COT 체인 결과를 요약하는 체인 실행
summarize_chain = LLMChain(llm=chat, prompt=summarize_prompt)

# 요약 체인의 결과 출력
summarize_chain_result = summarize_chain.invoke(
    cot_chain_result["text"]
)
print(f'- cot + summarize: {summarize_chain_result['text']}\n')

# SimpleSequentialChain 객체 생성 (COT 체인과 요약 체인을 순차적으로 실행)
cot_summarize_chain = SimpleSequentialChain(chains=[cot_chain, summarize_chain])

# 최종적으로 COT 체인 + 요약 체인 실행
result = cot_summarize_chain.invoke(
    "나는 시장에서 사과 10개를 샀다. 누나에게 2개, 엄마에게 2개 주었다. 그 다음 사과를 4개 더 사서 2개를 먹었다. 남은 갯수는 ?"
)

# 최종 결과 출력
print(f'- 최종 답변: {result["output"]}')

# llm 답변

"""
- cot: 단계별로 문제를 해결해 보겠습니다.

1. **처음 사과 개수**: 시장에서 사과 10개를 샀습니다.
   - 현재 사과 개수: 10개

2. **누나와 엄마에게 사과 주기**: 누나에게 2개, 엄마에게 2개를 주었습니다.        
   - 누나에게 준 사과: 2개
   - 엄마에게 준 사과: 2개
   - 총 준 사과: 2 + 2 = 4개
   - 남은 사과 개수: 10 - 4 = 6개

3. **사과 추가 구매**: 그 다음 사과를 4개 더 샀습니다.
   - 남은 사과 개수: 6개
   - 추가 구매한 사과: 4개
   - 현재 사과 개수: 6 + 4 = 10개

4. **사과 먹기**: 2개를 먹었습니다.
   - 현재 사과 개수: 10 - 2 = 8개

따라서, 남은 사과의 개수는 **8개**입니다.

- cot + summarize: 사과를 처음 10개 샀고, 누나와 엄마에게 각각 2개씩 주고 4개를 더
 사서 총 10개가 되었습니다. 이후 2개를 먹어 남은 사과는 8개입니다.

- 최종 답변: 처음 10개의 사과 중 4개를 누나와 엄마에게 주고, 4개를 추가로 사서 총 
10개가 되었으며, 2개를 먹은 후 남은 사과는 8개입니다.
"""