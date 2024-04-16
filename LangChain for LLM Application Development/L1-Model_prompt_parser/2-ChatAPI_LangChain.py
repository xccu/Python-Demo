#Chat API : LangChain
from langchain.prompts import ChatPromptTemplate

from utils.jwt_token import generate_zhipu_token
from langchain_openai import ChatOpenAI

# 目标：把给定的信息转化为自定义风格的信息
chat = model = ChatOpenAI(
    model_name= "glm-4",
    openai_api_base= "https://open.bigmodel.cn/api/paas/v4",
    openai_api_key=generate_zhipu_token(),
    streaming=False,
    verbose=True
)



#指定提示：将指定文本{text}转换为给定样式{style}
template_string = """Translate the text that is delimited by triple backticks \
into a style that is {style}. text: ```{text}```"""

#提示模板：在可能的情况下可复用的抽象
prompt_template = ChatPromptTemplate.from_template(template_string)
prompt_template.messages[0].prompt
prompt_template.messages[0].prompt.input_variables


#案例1

#自定义风格（平静的美式英语）
#customer_style = """American English in a calm and respectful tone"""
#自定义风格（平静的繁体中文）
customer_style = """Traditional Chinese in a calm and respectful tone"""

#待翻译的文本（海盗英文）
customer_email = """
Arrr, I be fuming that me blender lid flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help right now, matey!"""

customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)

#输出结果： <class 'list'>
print(type(customer_messages))

#输出结果：将英语海盗语风格的文本转化为平静的美式英语
print(customer_messages[0])

# Call the LLM to translate to the style of the customer message
# 调用LLM，将其翻译成自定义信息的风格
customer_response = chat(customer_messages)

print(customer_response.content)





#案例2
#待翻译的文本（海盗英文）
service_reply = """Hey there customer, the warranty does not cover \
cleaning expenses for your kitchen because it's your fault that \
you misused your blender by forgetting to put the lid on before \
starting the blender. Tough luck! See ya!"""

#风格设置为礼貌的海盗英文
#service_style = """a polite tone that speaks in English Pirate"""

#风格设置为礼貌的简体中文
service_style = """a polite tone that speaks in Simplified Chinese"""

service_messages = prompt_template.format_messages(
    style=service_style,
    text=service_reply)

print(service_messages[0].content)

service_response = chat(service_messages)
print(service_response.content)