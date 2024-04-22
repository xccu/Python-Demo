# import warnings
#
#
#
# warnings.filterwarnings('ignore')
#


from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate    #聊天提示模板
from langchain.chains import LLMChain               #LLM链
from utils.jwt_token import generate_zhipu_token

#加载本地环境变量文件 .env
# from utils import load_env
# load_env.load()

#LLM
llm = ChatOpenAI(
    model_name= "glm-4",
    openai_api_base= "https://open.bigmodel.cn/api/paas/v4",
    openai_api_key=generate_zhipu_token()
)

#提示器
#"最好的名字描述一家生产{product}的公司是什么?"
prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
)

#合成一个链（LLM+提示器组合）
chain = LLMChain(llm=llm, prompt=prompt)
# "Queen Size Sheet Set"/"大号床单套装"
product = "Queen Size Sheet Set"
result = chain.run(product)
print(result)