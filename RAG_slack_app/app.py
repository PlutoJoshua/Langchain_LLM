import os
import re
import time
import logging
from typing import Any
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from add_document import initialize_vectorstore

logging.basicConfig(
    filename='log.log',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding='utf-8'
)

# 챗봇 업데이트 간격 설정
CHAT_UPDATE_INTERVAL_SEC = 0.5

# .env 파일 로드
env_path = ".../.env"
load_dotenv(env_path)

# 환경변수 키 설정
api_key = os.getenv("OPENAI_API_KEY")
api_signing = os.getenv("SLACK_SIGNING_SECRET")
api_bot = os.getenv("SLACK_BOT_TOKEN")

# slack 초기화
app = App(
    signing_secret=api_signing, # 서명 키
    token=api_bot, # 봇 토큰
    process_before_response=True, # 전처리 활성화
)

# 대화 기록을 저장할 딕셔너리 (사용자별로 기록)
user_histories = {}

# Slack에서 스트리밍된 메시지의 처리를 위한 콜백 핸들러 정의
class SlackStreamingCallbackHandler(BaseCallbackHandler):
    last_send_time = time.time() # 마지막 메시지 전송 시간
    message = "" # 메시지 초기화

    def __init__(self, channel, ts):
        self.channel = channel # Slack 채널 ID
        self.ts = ts # 메시지 타임스탬프 (스레드 ID와 연결됨)
        self.interval = CHAT_UPDATE_INTERVAL_SEC # 메시지 전송 간격
        self.update_count = 0 # 메시지 업데이트 카운트

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # 새로운 토큰이 생성될 때마다 메시지에 추가
        self.message += token

        # 일정 시간 간격이 지나면 메시지를 업데이트하여 전송
        now = time.time()
        if now - self.last_send_time > self.interval:
            # Slack에 메시지 업데이트 (타이핑 중... 표시)
            app.client.chat_update(
                channel=self.channel, ts=self.ts, text=f'{self.message}\n\nTyping...'
            )
            self.last_send_time = now # 마지막 전송 시간을 갱신
            self.update_count += 1 # 업데이트 카운트 증가

            # 업데이트 속도 조절: 일정 수의 토큰마다 간격을 늘림
            if self.update_count / 10 > self.interval:
                self.interval = self.interval * 2

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        # LLM 응답이 끝나면 호출되는 함수
        message_context = "OpenAI API에서 생성되는 정보는 부정확하거나 부적절할 수 있으며, 우리의 견해를 나타내지 않습니다."
        
        # LLMResult에서 생성된 메시지 추출
        if response.generations and len(response.generations) > 0:
            self.message = ''.join([gen.text for gen in response.generations[0]])  # 첫 번째 생성된 텍스트를 가져옴
        else:
            self.message = "No message generated."  # 기본 메시지 설정

        # Slack에 메시지와 블록을 업데이트
        message_blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": self.message}},  # 텍스트 내용
            {"type": "divider"}, # 구분선
            {"type": "context",  # 컨텍스트 영역
            "elements": [{"type": "mrkdwn", "text": message_context}],
            },
        ]
        
        # 메시지 및 블록을 Slack 채널에 업데이트
        app.client.chat_update(
            channel=self.channel,
            ts=self.ts,
            text=self.message, # text 필드에 메시지 내용 추가
            blocks=message_blocks # blocks 필드에 추가적인 블록 내용 추가
        )

def format_docs(docs):
    logging.info("Retrieved documents: " + "\n\n".join(doc.page_content for doc in docs))
    return "\n\n".join(doc.page_content for doc in docs)

# Slack에서 멘션이 발생하면 호출되는 이벤트 처리 함수
def handle_mention(event, say):
    global user_histories  # 전역 변수를 사용
    # 이벤트에서 채널과 타임스탬프 가져오기
    channel = event["channel"]
    thread_ts=event["ts"]
    user = event["user"]  # 메시지 보낸 사용자 ID 가져오기
    message = re.sub("<@.*>", "", event["text"])  # 멘션 제거한 사용자 메시지 추출
    logging.info(f"User {user}: {message}")
    # 사용자별로 대화 기록이 없으면 새로 생성
    if user not in user_histories:
        user_histories[user] = []

    # 해당 사용자의 대화 기록
    history = user_histories[user]

    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever()

    # Retriever에서 검색
    relevant_docs = retriever.invoke(message)
    formatted_context = format_docs(relevant_docs)

    rephrase_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),  # Placeholder for chat history
            ("user", "{input}"),  # User input message
            ("user", "위의 대화에서 대화와 관련된 정보를 찾기 위한 검색 쿼리를 생성해 주세요."),  # Prompt for query generation
        ]
    )

    # Slack에 "타이핑 중..." 메시지 전송
    result = say("\n\nTyping...", user=event["user"], thread_ts=thread_ts)
    ts = result["ts"] # 새로운 메시지 타임스탬프

    # SlackStreamingCallbackHandler를 먼저 생성
    callback = SlackStreamingCallbackHandler(channel=channel, ts=ts)
    # 대화 기록을 Langchain 메시지 형식으로 변환
    chat_history = []
    for entry in history:
        if entry['role'] == 'user':
            chat_history.append(HumanMessage(content=entry['content']))
        elif entry['role'] == 'ai':
            chat_history.append(AIMessage(content=entry['content']))
   
    # ChatOpenAI를 통해 모델 호출 (스트리밍 활성화)
    rephrase_llm = ChatOpenAI(
        model_name='gpt-4o-mini',
        temperature=0,
        streaming=True,
        callbacks=[callback]
    )

    rephrase_chain = create_history_aware_retriever(
        rephrase_llm, retriever, rephrase_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "아래의 문맥만을 고려하여 질문에 답하세요. \n\n{context}"),  # System prompt
            (MessagesPlaceholder(variable_name="chat_history")),  # Chat history placeholder
            ("user", "{input}"),  # User input
        ]
    )

    # ChatOpenAI를 통해 모델 호출 (스트리밍 활성화)
    qa_llm = ChatOpenAI(
        model_name='gpt-4o-mini',
        temperature=0,
        streaming=True,
        callbacks=[callback]
    )

        # chain 생성
    qa_chain = qa_prompt | qa_llm | StrOutputParser()

    conversational_retriever_chain = (
        RunnablePassthrough.assign(context=rephrase_chain | format_docs) | qa_chain
    )
    # AI 응답 생성
    ai_message = conversational_retriever_chain.invoke(
        {
            "input": message,
            "context": formatted_context,
            "chat_history": chat_history,
        }
    )
    logging.info(f"AI massage: {ai_message}")
    logging.info(f"chat_history: {history}")
    # 사용자 메시지와 AI 메시지를 리스트에 추가
    history.append({"role": "user", "content": message})
    history.append({"role": "ai", "content": ai_message})

    logging.info(f"Updated history for thread {user}: {history}")

# 이벤트 처리: "app_mention" 이벤트 발생 시 just_ack 함수 호출 후 handle_mention 호출
def just_ack(ack):
    ack()
    
# Slack에서 앱 멘션 이벤트 처리
app.event("app_mention")(ack=just_ack, lazy=[handle_mention])

# 앱 실행 (SocketModeHandler를 사용하여 슬랙 앱을 실행)
if __name__ == "__main__":
    SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN")).start()