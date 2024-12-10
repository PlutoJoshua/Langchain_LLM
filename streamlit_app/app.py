import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent, load_tools
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# .env 파일 로드
env_path = ".../.env"
load_dotenv(env_path)

# OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY")

# agent chain 생성 함수 정의
def create_agent_chain(history):
    chat = ChatOpenAI(
        model_name='gpt-4o-mini',
        temperature=0
    )
    # tools 로드(검색엔진(duckduckgo, wikipedia))
    tools = load_tools(['ddg-search', 'wikipedia'])
    # langchain hub에서 프롬프트 불러오기
    prompt = hub.pull("hwchase17/openai-tools-agent")
    # 대화 메모리 설정
    memory = ConversationBufferMemory(
        chat_memory = history, memory_key="chat_history", return_messages=True
    )
    # agnet 실행
    agent = create_openai_tools_agent(chat, tools, prompt)
    # agentexecutor 반환
    return AgentExecutor(agent=agent, tools=tools, memory=memory)

### streamlit 설정
st.title("LangChain-streamlit-app")

# 대화 기록을 위한 객체 생성
history = StreamlitChatMessageHistory()

# 대화 메시지 기록이 있으면 화면에 표시
for message in history.messages:
    st.chat_message(message.type).write(message.content)

# 사용자 입력 받기
prompt = st.chat_input("What is up?")

if prompt:
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)

    # llm 응답을 생성하고 표시
    with st.chat_message("assistant"):
        # streamlit callback handler 표시(진행 상황 표시)
        callback = StreamlitCallbackHandler(st.container())
        # agent chain 생성
        agent_chain = create_agent_chain(history)
        # agent chain에 사용자 입력을 전달하고 응답 받기
        response = agent_chain.invoke(
            {"input": prompt},
            {"callbacks": [callback]}
        )

        # 답변 출력
        st.markdown(response["output"])