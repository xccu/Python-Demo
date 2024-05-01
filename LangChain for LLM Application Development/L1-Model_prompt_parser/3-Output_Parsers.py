# Let's start with defining how we would like the LLM output to look like:
import json

from langchain_community.chat_models import ChatZhipuAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from utils import load_env
from utils.jwt_token import generate_zhipu_token

#让llm输出json，并使用langchain解析该输出
#从文本中提取信息，并输出为json格式

#希望输出的格式
# gift = {
#   "gift": False,
#   "delivery_days": 5,
#   "price_value": "pretty affordable!"
# }

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

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""

# read local .env file
load_env.load()

#提示模板
prompt_template = ChatPromptTemplate.from_template(review_template)
print(prompt_template)

messages = prompt_template.format_messages(text=customer_review)

# 目标：把给定的信息转化为自定义风格的信息
chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.5,
)

response = chat(messages)
print(response.content)

type(response.content)

# You will get an error by running this line of code
# because'gift' is not a dictionary
# 'gift' is a string
#str = response.content.get('gift')

json_obj = json.loads(response.content).get('gift')
print(json_obj)