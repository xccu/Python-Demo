from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from utils.jwt_token import generate_zhipu_token

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

llm = ChatOpenAI(
    model_name= "glm-4",
    openai_api_base= "https://open.bigmodel.cn/api/paas/v4",
    openai_api_key=generate_zhipu_token(),
    verbose=True
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