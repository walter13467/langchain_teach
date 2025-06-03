import glob, json
from langchain.schema import Document

docs = []
for fn in glob.glob("laws/*.json"):
    with open(fn, encoding="utf-8-sig") as f:
        law_data = json.load(f)
    # chlaw.json格式中，法規列表在"Laws"鍵中
    if "Laws" in law_data:
        for law in law_data["Laws"]:
            title = law.get("LawName", "")
            law_url  = law["LawURL"]
            # 條文列表在"LawArticles"鍵中
            for article in law.get("LawArticles", []):
                content = article.get("ArticleContent", "")
                article_no = article.get("ArticleNo", "")
                meta = {
                    "title": title,
                    "article_no": article_no,
                    "law_url": law_url
                }
                docs.append(Document(page_content=content, metadata=meta))



#向量
#dotenv
import os
from dotenv import load_dotenv
load_dotenv()

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

EMB = OpenAIEmbeddings(model="text-embedding-3-small",
                        openai_api_key=os.getenv("OPENAI_API_KEY"))
vector_store = FAISS.from_documents(docs, EMB)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


#LLM
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import RetrievalQA


# 建立Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0)

# 建立RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


# 試問答
import re

def convert_url(url):
    """
    將 LawAll.aspx?pcode=…&flno=… 的網址
    轉成 LawSingle.aspx?pcode=…&flno=…
    """
    pattern = r"(LawClass/)LawAll\.aspx\?pcode=([^&]+)&flno=(\d+)"
    repl    = r"\1LawSingle.aspx?pcode=\2&flno=\3"
    return re.sub(pattern, repl, url)

query = "幾歲可以選總統?"
res = qa({"query": query})
print("▶ 回答： ", res["result"], "\n")
print("▶ 依據條文：\n")
for doc in res["source_documents"]:
    print(f" - {doc.metadata['title']} {doc.metadata['article_no']}")
    print(f"   {doc.page_content}\n")
    articleNo = re.sub(r"\D", "", doc.metadata['article_no'])
    orig_url = f"{doc.metadata['law_url']}&flno={articleNo}"
    url =  convert_url(orig_url)
    print("       條文網址：")
    print("       "+url+"\n")

