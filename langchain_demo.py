#pip install langchain-openai
#pip install langchain

# https://blog.csdn.net/oHeHui1/article/details/136389922

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from configure import get_yaml
from utils.jwt_token import generate_token

key = get_yaml('zhipu.key')
exp = get_yaml('zhipu.exp')

token = generate_token(key,exp)

zhipu_llm = ChatOpenAI(
    model_name= "glm-4",
    openai_api_base= "https://open.bigmodel.cn/api/paas/v4",
    openai_api_key=token,
    streaming=False,
    verbose=True
)

messages = [
     # AIMessage(content="Hi."),
     # SystemMessage(content="Your role is a poet."),
     HumanMessage(content="深圳2008年的GDP多少亿"),
     #HumanMessage(content="only give me the result,no other words:the result of add 3 to 4"),
 ]

response = zhipu_llm.invoke(messages)
print(response)


#zhipu_llm.streaming = True
#for chunk in zhipu_llm.stream("猪八戒的爸爸是谁"):
    # print(chunk.content, end="", flush=True)
#    print(chunk.content)
