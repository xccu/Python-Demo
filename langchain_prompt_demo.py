#https://python.langchain.com/docs/expression_language/get_started/

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from configure import get_yaml
from utils.jwt_token import generate_token

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
prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

print("print prompt:")
prompt_value = prompt.invoke({"topic": "冰淇淋"})
print(prompt_value.to_messages())
print(prompt_value.to_string())

print("print response:")
response = chain.invoke({"topic": "冰淇淋"})
print(response)
