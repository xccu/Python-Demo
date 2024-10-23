#pip install langchain-openai
#pip install langchain

from configure import get_yaml
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

key = get_yaml('zhipu.key')

llm = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key=key,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "你是个聊天机器人，知道很多知识"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)
response = conversation.invoke({"question": "智普AI如何读取txt文件并生成摘要，请给出示例代码"})
print(response)