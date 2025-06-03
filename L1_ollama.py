import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama

# 使用langchain的google實例
llm = Ollama(model='gemma3:4b')

# 建構prompt內容
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
# 將prompt傳入到模型
ai_msg = llm.invoke(messages)

# 顯示內容
print(ai_msg)