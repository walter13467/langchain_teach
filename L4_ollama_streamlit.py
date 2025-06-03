import os
import re
import glob, json
import streamlit as st

from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from langchain_community.llms import Ollama

#dotenv
from dotenv import load_dotenv
load_dotenv()


st.set_page_config(page_title="法律 RAG ")
st.title("全國法規資料庫 RAG + Gemini 問答")

# 快取：建立向量庫與 RetrievalQA Chain
@st.cache_resource(show_spinner=False)
def init_qa_chain():
    docs = []
    for fn in glob.glob("laws/*.json"):
        with open(fn, encoding="utf-8-sig") as f:
            law_data = json.load(f)
        for law in law_data.get("Laws", []):
            title   = law.get("LawName", "")
            law_url = law.get("LawURL", "")
            for art in law.get("LawArticles", []):
                content    = art.get("ArticleContent", "")
                article_no = art.get("ArticleNo", "")
                meta = {
                    "title": title,
                    "article_no": article_no,
                    "law_url": law_url
                }
                docs.append(Document(page_content=content, metadata=meta))

    # Embedding + FAISS
    emb = OpenAIEmbeddings(
                        model="text-embedding-3-small",
                        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    vector_store = FAISS.from_documents(docs, emb)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 設定 Ollama
    llm = Ollama(model='gemma3:4b')

    # RetrievalQA Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

# 載入 QA Chain
qa_chain = init_qa_chain()

# 使用者介面
query = st.text_input("請輸入法律問題：")
if st.button("開始查詢") and query:
    with st.spinner("系統搜尋中，請稍候…"):
        res = qa_chain({"query": query})

    # 顯示 LLM 回答
    st.subheader("▶ 回答")
    st.write(res["result"])

    
    #轉換網址
    def convert_url(url):
        pattern = r"(LawClass/)LawAll\.aspx\?pcode=([^&]+)&flno=(\d+)"
        repl    = r"\1LawSingle.aspx?pcode=\2&flno=\3"
        return re.sub(pattern, repl, url)
    
    # 顯示依據條文與url
    st.subheader("▶ 依據條文")
    for doc in res["source_documents"]:
        title = doc.metadata.get("title","")
        artno = doc.metadata.get("article_no","")
        url   = doc.metadata.get("law_url","")
        snippet = doc.page_content.strip().replace("\n"," ")

        articleNo = re.sub(r"\D", "", doc.metadata['article_no'])
        orig_url = f"{doc.metadata['law_url']}&flno={articleNo}"
        url =  convert_url(orig_url)
        # 顯示條號、片段、url
        st.markdown(
            f"- **{title} {artno}**  "
            f"    [查看法條全文]({url})  \n"
            f"> …{snippet[:100]}…"
        )
