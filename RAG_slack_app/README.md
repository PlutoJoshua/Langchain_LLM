# 슬랙 챗봇 앱 구축하기(RAG)
- 책에서는 Cloud9과 LAmbda, memento를 사용하지만 코드 내에서만 저장하게 구현하였습니다.
- data는 kocw(대학 공개 강의)의 김준영 교수님의 강의 자료입니다.
 - http://www.kocw.net/home/cview.do?cid=da6602c011280fea
- vectorstore : pinecone
- event의 유저 정보로 대화를 저장합니다.
- history : 딕셔너리로 직접 message와 ai_message를 추가하여 구성
- logging 구현
- 기본적인 틀은 같은 폴더에 있는 slack_app과 동일합니다.
- 자세한 제작 과정은 velog에 업로드 예정

## app process
![](https://github.com/PlutoJoshua/Langchain_LLM/blob/main/photo/process.png?raw=true)

## question
![](https://github.com/PlutoJoshua/Langchain_LLM/blob/main/photo/rag1.png?raw=true)

## documents_search
![](https://github.com/PlutoJoshua/Langchain_LLM/blob/main/photo/rag2.png?raw=true)
![](https://github.com/PlutoJoshua/Langchain_LLM/blob/main/photo/rag3.png?raw=true)