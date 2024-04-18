# https://learn.deeplearning.ai/courses/langchain/lesson/3/memory

import os
from dotenv import load_dotenv, find_dotenv

# _ = load_dotenv(find_dotenv()) # read local .env file
#
# import warnings
# warnings.filterwarnings('ignore')

from utils.jwt_token import generate_zhipu_token

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(
    model_name= "glm-4",
    openai_api_base= "https://open.bigmodel.cn/api/paas/v4",
    openai_api_key=generate_zhipu_token()
    #verbose=True #显示提示
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