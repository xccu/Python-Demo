from configure import get_yaml
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

#让llm编写一份长文本的摘要

#目前仅gpt模型支持
# llm = ChatOpenAI(
#     model_name= get_yaml('gpt.model'),
#     openai_api_base=  get_yaml('gpt.api_base'),
#     openai_api_key= get_yaml('gpt.key'),
#     verbose=True
# )
from utils import load_env

load_env.load()
llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"},
                    {"output": f"{schedule}"})

print(memory.load_memory_variables({}))

conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=True
)

conversation.predict(input="What would be a good demo to show?")

print(memory.load_memory_variables({}))