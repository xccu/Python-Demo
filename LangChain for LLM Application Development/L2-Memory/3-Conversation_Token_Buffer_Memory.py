# pip install tiktoken
#会话token缓存，将限制内存保存的token数量

from langchain.memory import ConversationTokenBufferMemory
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

from configure import get_yaml
#目前仅gpt模型支持
llm = ChatOpenAI(
    model_name= get_yaml('gpt.model'),
    openai_api_base=  get_yaml('gpt.api_base'),
    openai_api_key= get_yaml('gpt.key'),
    verbose=True
)

#限制token数为50
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"},
                    {"output": "Charming!"})

print(memory.load_memory_variables({}))