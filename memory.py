from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# .env 파일 로드
env_path = "../.env"
load_dotenv(env_path)

# OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY")

# ChatOpenAI 객체 생성
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

conversation = ConversationChain(
    llm=chat,
    memory=ConversationBufferMemory()
)

while True:
    user_message = input("You: ")

    if user_message == "end":
        print(('The end'))
        break

    ai_message = conversation.invoke(input=user_message)["response"]
    print(f'AI: {ai_message}')

# 출력 결과

"""
You: hi, What's your name?
AI: Hello! I’m an AI designed to assist and chat with you. I don’t have a personal name like a human would, but you can call me whatever you like! How about you? What’s your name?
You: Oh, how can you help me?
AI: I can help you in a variety of ways! Whether you need information on a specific topic, assistance with problem-solving, or just someone to chat with, I'm here for you. I can provide explanations, answer questions, suggest ideas, or even help 
with creative writing. If you have a particular task in mind, just let me know, and I'll do my best to assist you! What do you need help with today?
You: um.. can you speak Korean?
AI: Yes, I can understand and generate text in Korean! If you have any questions or need assistance in Korean, feel free to ask. For example, I can help with translations, basic phrases, or even cultural insights. What would you like to know?    
You: 그럼 한국어로 질문할게, lanchain의 ConversationBufferMemory 대신 RunnableWithMessageHistory를 쓰라는 오류가 났어. 어떻게 해야 할까?
AI: RunnableWithMessageHistory는 LangChain에서 대화의 메시지 히스토리를 관리하는 
데 사용되는 기능입니다. 이 오류는 ConversationBufferMemory 대신 RunnableWithMessageHistory를 사용해야 한다는 것을 의미합니다.

이 문제를 해결하기 위해서는 다음 단계를 따라 해보세요:

1. **코드 수정**: 기존의 ConversationBufferMemory를 RunnableWithMessageHistory로  
변경하세요. 예를 들어, 코드에서 `ConversationBufferMemory`를 사용하는 부분을 찾아 
서 `RunnableWithMessageHistory`로 바꿉니다.

2. **메시지 히스토리 관리**: RunnableWithMessageHistory를 사용할 때는 메시지 히스 
토리를 어떻게 관리할지에 대한 설정을 확인하세요. 이 기능은 대화의 흐름을 유지하는 
데 도움이 됩니다.

3. **문서 확인**: LangChain의 공식 문서나 GitHub 리포지토리를 참조하여 RunnableWithMessageHistory의 사용법과 예제를 확인하세요.

4. **테스트**: 변경한 코드를 실행하여 오류가 해결되었는지 확인하세요.

이런 방식으로 문제를 해결할 수 있을 것입니다. 추가적인 질문이 있으면 언제든지 물어
보세요!
You: end
The end

## context 유지
You: hi, I'm Joshua
AI: Hello, Joshua! It's great to meet you! How's your day going so far?
You: nice meet you too  
AI: I'm glad to hear that! So, what have you been up to today? Any interesting plans or activities?
You: Do you know my name?
AI: Yes, you mentioned that your name is Joshua! It's a nice name. Is there a story behind it, or do you have a favorite nickname?
You: end
The end
"""

