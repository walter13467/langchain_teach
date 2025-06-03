import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=api_key,
)

# 標題
st.title('基礎streamlit + Gemini')

with st.form('form_1'):
    text = st.text_area('Enter text:', '')   # 文字輸入
    submitted = st.form_submit_button('送出')
    if submitted:
        st.write(llm.invoke(text).content)  # 文字輸出