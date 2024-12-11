import json
import logging
import os
import re
import time
from typing import Any
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

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

# 대화 기록을 저장할 딕셔너리
thread_histories = {}

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

# Slack에서 멘션이 발생하면 호출되는 이벤트 처리 함수
def handle_mention(event, say):
    # 이벤트에서 채널과 타임스탬프 가져오기
    channel = event["channel"]
    thread_ts = event["ts"]

    # 메시지에서 멘션을 제거하여 사용자 입력만 추출
    message = re.sub("<@.*>", "", event["text"])

    # 스레드 기록이 없으면 새로 생성
    if thread_ts not in thread_histories:
        thread_histories[thread_ts] = []

    # 해당 스레드의 대화 기록
    memory = thread_histories[thread_ts]

    # Slack에 "타이핑 중..." 메시지 전송
    result = say("\n\nTyping...", thread_ts=thread_ts)
    ts = result["ts"] # 새로운 메시지 타임스탬프

    # LLM에 전달할 프롬프트 템플릿 생성
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are good assistant."), # system prompt
            (MessagesPlaceholder(variable_name="chat_history")), # 대화 기록
            ("user", "{input}") # user message
        ]
    )

    # SlackStreamingCallbackHandler를 통해 메시지 스트리밍
    callback = SlackStreamingCallbackHandler(channel=channel, ts=ts)

    # ChatOpenAI를 통해 모델 호출 (스트리밍 활성화)
    llm = ChatOpenAI(
        model_name='gpt-4o-mini',
        temperature=0,
        streaming=True,
        callbacks=[callback]
    )

    # chain 생성
    chain = prompt | llm | StrOutputParser()

    # AI의 응답 생성
    ai_message = chain.invoke({"input": message, "chat_history": memory})

    # 사용자 메시지와 AI 메시지를 리스트에 추가
    memory.append({"role": "user", "content": message})
    memory.append({"role": "ai", "content": ai_message})

# 이벤트 처리: "app_mention" 이벤트 발생 시 just_ack 함수 호출 후 handle_mention 호출
def just_ack(ack):
    ack()
    
# Slack에서 앱 멘션 이벤트 처리
app.event("app_mention")(ack=just_ack, lazy=[handle_mention])

# 앱 실행 (SocketModeHandler를 사용하여 슬랙 앱을 실행)
if __name__ == "__main__":
    SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN")).start()