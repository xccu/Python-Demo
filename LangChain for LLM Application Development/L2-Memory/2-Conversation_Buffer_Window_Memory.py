from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatZhipuAI

from utils import load_env

#只保留一个窗口记忆的对话缓存记忆

#变量k表示设置记住的对话条数
memory = ConversationBufferWindowMemory(k=1)
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
# 只记住了最近一次对话
print(memory.buffer)
print(memory.load_memory_variables({}))

# 加载本地 .env 文件
load_env.load()
# 目标：把给定的信息转化为自定义风格的信息
llm = ChatZhipuAI(
    model="glm-4",
    temperature=0.5,
)

#变量k表示设置记住的对话条数
memory = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=False
)

conversation.predict(input="Hi, my name is Andrew")
conversation.predict(input="What is 1+1?")
conversation.predict(input="What is my name?")
# my name is Andrew无法被记住
print(memory.buffer)
print(memory.load_memory_variables({}))