# Parse the LLM output string into a Python dictionary

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from utils.jwt_token import generate_zhipu_token

#让llm输出json，并使用langchain解析该输出

#langchain响应模式
gift_schema = ResponseSchema(
    name="gift",
    description="Was the item purchased as a gift for someone else? \
    Answer True if yes,False if not or unknown.")

delivery_days_schema = ResponseSchema(
    name="delivery_days",
    description="How many days did it take for the product to arrive?\
    If this information is not found,output -1.")

price_value_schema = ResponseSchema(
    name="price_value",
    description="Extract any sentences about the value or price, \
    and output them as a comma separated Python list.")

response_schemas = [gift_schema, delivery_days_schema, price_value_schema]
#使用langchain结构化输出解析器
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()

print(format_instructions)

#文本
customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""
#提示模板（问答形式提取需要的信息）
review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

#提示模板
prompt = ChatPromptTemplate.from_template(template=review_template)

messages = prompt.format_messages(text=customer_review,
                                format_instructions=format_instructions)

print(messages[0].content)

chat = ChatOpenAI(
    model_name= "glm-4",
    openai_api_base= "https://open.bigmodel.cn/api/paas/v4",
    openai_api_key= generate_zhipu_token(),
    streaming=False,
    verbose=True
)
response = chat(messages)

print(response.content)

output_dict = output_parser.parse(response.content)

#output_dict
#type(output_dict)
print("delivery_days="+output_dict.get('delivery_days'))