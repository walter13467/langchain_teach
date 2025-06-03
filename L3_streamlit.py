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

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

# 組合成chain
chain = prompt | llm


# streamlit
st.title('翻譯')

LANGUAGES = [
    "English",
    "Traditional Chinese",
    "Simplified Chinese",
    "Japanese",
    "German",
    "French",
    "Spanish"
]

input_text = st.text_area("輸入文字", height=150)
col1, col2 = st.columns(2)
with col1:
    input_lang = st.selectbox("來源語言", LANGUAGES, index=0)
with col2:
    output_lang = st.selectbox("目標語言", LANGUAGES, index=1)

if st.button("開始翻譯"):
    if not input_text.strip():
        st.warning("請先輸入文字再進行翻譯。")
    else:
        ai_msg = chain.invoke(
            {
            "input_language":input_lang,
            "output_language":output_lang,
            "input":input_text
            }
        )
        st.write(ai_msg.content)
