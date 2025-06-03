import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# 提取api key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


# 使用langchain的google實例
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=api_key,
)

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
print(ai_msg.content)
