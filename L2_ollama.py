import streamlit as st
from langchain_community.llms import Ollama


llm = Ollama(model='gemma3:4b')

# 標題
st.title('基礎streamlit + Gemini')

with st.form('form_1'):
    text = st.text_area('Enter text:', '')   # 文字輸入
    submitted = st.form_submit_button('送出')
    if submitted:
        st.write(llm.invoke(text))  # 文字輸出
