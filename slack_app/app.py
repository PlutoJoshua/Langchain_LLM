import json
import logging
import os
import re
import time
from datetime import timedelta
from typing import Any
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

CHAT_UPDATE_INTERVAL_SEC = 1

# .env 파일 로드
env_path = ".../.env"
load_dotenv(env_path)

# OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY")
api_signing = os.getenv("SLACK_SIGNING_SECRET")
api_bot = os.getenv("SLACK_BOT_TOKEN")

app = App(
    signing_secret=api_signing,
    token=api_bot,
    process_before_response=True,
)

thread_histories = {}

class SlackStreamingCallbackHandler(BaseCallbackHandler):
    last_send_time = time.time()
    message = ""

    def __init__(self, channel, ts):
        self.channel = channel
        self.ts = ts
        self.interval = CHAT_UPDATE_INTERVAL_SEC

        self.update_count = 0

    def om_llm_new_token(self, token: str, **kwargs) -> None:
        self.message += token
        now = time.time()
        if now - self.last_send_time > self.interval:
            app.client.chat_update(
                channel=self.channel, ts=self.ts, text=f'{self.message}\n\nTyping...'
            )
            self.last._send_time = now
            self.update_count += 1

            if self.update_count / 10 > self.interval:
                self.interval = self.interval * 2

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        message_context = "OpenAI API에서 생성되는 정보는 부정확하거나 부적절할 수 있으며, 우리의 견해를 나타내지 않습니다."
        message_blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": self.message}},
            {"type": "divider"},
            {"type":"context",
             "elements": [{"type": "mrkdwn", "text": message_context}],
             },
        ]
        app.client.chat_update(
            channel=self.channel,
            ts=self.ts,
            text=self.message,
            blocks=message_blocks
        )

def handle_mention(event, say):
    channel = event["channel"]
    thread_ts = event["ts"]
    message = re.sub("<@.*>", "", event["text"])

    id_ts = event["ts"]
    if "thread_ts" not in thread_histories:
        thread_histories["tread_ts"] = ConversationBufferMemory

    memory = ConversationBufferMemory()

    result = say("\n\nTyping...", thread_ts=thread_ts)
    ts = result["ts"]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are good assistant."),
            (MessagesPlaceholder(variable_neame="chat_history")),
            ("user", "{input}")
        ]
    )

    callback = SlackStreamingCallbackHandler(channel=channel, ts=ts)

    llm = ChatOpenAI(
        model_name='gpt-4o-mini',
        temperature=0,
        streaming=True,
        callbacks=[callback]
    )

    chain = prompt | llm | StrOutputParser()

    ai_message = chain.invoke({"input": message, "chat_history": memory.chat_memory.messages})

    memory.add_user_message(message)
    memory.add_ai_message(ai_message)

def just_ack(ack):
    ack()
    
app.event("app_mention")(ack=just_ack, lazy=[handle_mention])

if __name__ == "__main__":
    SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN")).start()