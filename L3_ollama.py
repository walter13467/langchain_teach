from langchain_community.llms import Ollama

llm = Ollama(model='gemma3:4b')


from langchain_core.prompts import ChatPromptTemplate

# prompt的部分
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("assistant", "{input}"),
    ]
)

# 組合成chain
chain = prompt | llm

# 這裡可以改不同輸入與輸出語言以及輸入的文字
ai_msg = chain.invoke(
    {
        "input_language": "English",
        "output_language": "Traditional Chinese",
        "input": "I love programming.",
    }
)

print(ai_msg)
