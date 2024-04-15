# https://python.langchain.com/docs/expression_language/get_started/#rag-search-example
# 运行一个检索增强生成链，以便在回答问题时添加一些上下文
# pip install -qU langchain-openai

import getpass
import os

from langchain_openai import ChatOpenAI

from configure import get_yaml
from utils.jwt_token import generate_token
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings


# os.environ["OPENAI_API_KEY"] = getpass.getpass()

key = get_yaml('zhipu.key')
exp = get_yaml('zhipu.exp')

token = generate_token(key,exp)

model = ChatOpenAI(
    model_name= "glm-4",
    openai_api_base= "https://open.bigmodel.cn/api/paas/v4",
    openai_api_key=token,
    streaming=False,
    verbose=True
)



vectorstore = DocArrayInMemorySearch.from_texts(
    ["harrison worked at kensho", "bears like to eat honey"],
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
chain = setup_and_retrieval | prompt | model | output_parser

chain.invoke("where did harrison work?")