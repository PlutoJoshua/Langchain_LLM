from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
# use function calling
# from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# .env 파일 로드
env_path = "../.env"
load_dotenv(env_path)

# OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY")

# openai 인스턴스 생성
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# 사용할 도구 로드
tools = load_tools(["terminal"], allow_dangerous_tools=True)

# hub 에서 prompt 가져오기
prompt = hub.pull("hwchase17/react")

# react agent 생성
agent = create_react_agent(llm, tools, prompt)

# function calling
# agent = create_openai_functions_agent(llm, tools, prompt)
# agent 실행기 생성
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 결과 호출
result = agent_executor.invoke({"input": "현재 디렉토리에 있는 파일 목록을 알려줘"})
print(result["output"])

# 최종 답변

"""
Finished chain.
현재 디렉토리에 있는 파일 목록은 다음과 같습니다:
- .gitignore
- Agent.py
- chain-chain.py
- chain.py
- data
- faiss_index
- memory.py
- RAG.py
- README.md
"""