# https://learn.deeplearning.ai/courses/langchain/lesson/3/memory

import os
from dotenv import load_dotenv, find_dotenv

# _ = load_dotenv(find_dotenv()) # read local .env file
#
# import warnings
# warnings.filterwarnings('ignore')
from langchain_community.chat_models import ChatZhipuAI

from utils import load_env

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# 加载本地 .env 文件
load_env.load()
# 目标：把给定的信息转化为自定义风格的信息
llm = ChatZhipuAI(
    model="glm-4",
    temperature=0.5,
)
# memory变量用于储存记忆
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=True
)

conversation.predict(input="Hi, my name is Andrew")
conversation.predict(input="What is 1+1?")
conversation.predict(input="What is my name?")
# 打印Memory 即langchain在对话中记住的内容
print(memory.buffer)
print(memory.load_memory_variables({}))
#对话缓存
memory = ConversationBufferMemory()
#对话缓存指定输入输出
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
# Human: Hi
# AI: What's up
print(memory.buffer)
# 打印Memory 即langchain在对话中记住的内容
print(memory.load_memory_variables({}))
#继续添加额外的记忆
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
# 打印Memory 即langchain在对话中记住的内容
print(memory.load_memory_variables({}))