#顺序链

#pip install pandas
from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate    #聊天提示模板
from langchain.chains import LLMChain               #LLM链
from utils.jwt_token import generate_zhipu_token

import pandas as pd

df = pd.read_csv('Data.csv')
df.head()

llm = ChatOpenAI(
    model_name= "glm-4",
    openai_api_base= "https://open.bigmodel.cn/api/paas/v4",
    openai_api_key=generate_zhipu_token()
)
# prompt template 1: translate to english
# 评论翻译成英语
first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to english:\n\n{Review}"
)
# chain 1: input= Review and output= English_Review
chain_one = LLMChain(
    llm=llm,
    prompt=first_prompt,
    output_key="English_Review"
)

second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:\n\n{English_Review}"
)
# chain 2: input= English_Review and output= summary
# 用英语总结评论
chain_two = LLMChain(llm=llm, prompt=second_prompt,output_key="summary")

# prompt template 3: translate to english
third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review:\n\n{Review}"
)
# chain 3: input= Review and output= language
# 检测最初的评论语言
chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="language" )

# prompt template 4: follow up message
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
)
# chain 4: input= summary, language and output= followup_message
# 接受多个变量
chain_four = LLMChain(llm=llm, prompt=fourth_prompt,output_key="followup_message")

# overall_chain: input= Review
# and output= English_Review,summary, followup_message
# 组合成链
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["English_Review", "summary","followup_message"],
    verbose=True
)

review = df.Review[5]
result = overall_chain(review)
print(result)