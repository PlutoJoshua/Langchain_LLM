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

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

cot_template = """다음 질문에 답하세요.

질문: {question}

단계별로 생각해 봅시다.
"""

cot_prompt = PromptTemplate(
    input_variables=["question"],
    template=cot_template,
)

cot_chain = LLMChain(llm=chat, prompt=cot_prompt)

cot_chain_result = cot_chain.invoke(
    "나는 시장에서 사과 10개를 샀다. 누나에게 2개, 엄마에게 2개 주었다. 그 다음 사과를 4개 더 사서 2개를 먹었다. 남은 갯수는 ?"
)
print(f'- cot: {cot_chain_result['text']}\n')


summarize_template = """다음 문장을 결론만 간단히 요약하세요.

{input}
"""

summarize_prompt = PromptTemplate(
    input_variables=["input"],
    template=summarize_template
)

summarize_chain = LLMChain(llm=chat, prompt=summarize_prompt)

summarize_chain_result = cot_chain.invoke(
    "나는 시장에서 사과 10개를 샀다. 누나에게 2개, 엄마에게 2개 주었다. 그 다음 사과를 4개 더 사서 2개를 먹었다. 남은 갯수는 ?"
)
print(f'- cot + summarize: {summarize_chain_result['text']}\n')

cot_summarize_chain = SimpleSequentialChain(chains=[cot_chain, summarize_chain])

result = cot_summarize_chain.invoke(
    "나는 시장에서 사과 10개를 샀다. 누나에게 2개, 엄마에게 2개 주었다. 그 다음 사과를 4개 더 사서 2개를 먹었다. 남은 갯수는 ?"
)

print(f'- 최종 답변: {result["output"]}')

# llm 답변

"""
- cot: 단계별로 문제를 해결해 보겠습니다.

1. **처음 사과 개수**: 시장에서 사과 10개를 샀습니다.
   - 현재 사과 개수: 10개

2. **누나와 엄마에게 사과 주기**: 누나에게 2개, 엄마에게 2개를 주었습니다.        
   - 누나에게 준 사과: 2개
   - 엄마에게 준 사과: 2개
   - 총 준 사과: 2개 + 2개 = 4개
   - 남은 사과 개수: 10개 - 4개 = 6개

3. **사과 추가 구매**: 그 다음 사과를 4개 더 샀습니다.
   - 남은 사과 개수: 6개
   - 추가 구매한 사과: 4개
   - 총 사과 개수: 6개 + 4개 = 10개

4. **사과 먹기**: 2개를 먹었습니다.
   - 먹은 사과: 2개
   - 남은 사과 개수: 10개 - 2개 = 8개

결론적으로, 남은 사과의 개수는 **8개**입니다.

- cot + summarize: 단계별로 문제를 해결해 보겠습니다.

1. **처음 사과 개수**: 10개
2. **누나에게 준 사과**: 2개
   - 남은 사과: 10 - 2 = 8개
3. **엄마에게 준 사과**: 2개
   - 남은 사과: 8 - 2 = 6개
4. **추가로 사과 구매**: 4개
   - 현재 사과 개수: 6 + 4 = 10개
5. **먹은 사과**: 2개
   - 남은 사과: 10 - 2 = 8개

따라서, 남은 사과의 개수는 **8개**입니다.

- 최종 답변: 남은 사과의 개수는 **8개**입니다.

"""