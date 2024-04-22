#简单的顺序链，一个接一个运行链的序列

from langchain.chains import SimpleSequentialChain #简单顺序链
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from utils.jwt_token import generate_zhipu_token

#加载本地环境变量文件 .env
# from utils import load_env
# load_env.load()



llm = ChatOpenAI(
    model_name= "glm-4",
    openai_api_base= "https://open.bigmodel.cn/api/paas/v4",
    openai_api_key=generate_zhipu_token()
)

# 提示器1
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
)
# 链（chain）1接收产品并返回最佳公司名称
chain_one = LLMChain(llm=llm, prompt=first_prompt)

# 提示器2
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following company:{company_name}"
)

# 链（chain）2 接收公司名称，输出20个字的描述
chain_two = LLMChain(llm=llm, prompt=second_prompt)

overall_simple_chain = SimpleSequentialChain(
    chains=[chain_one, chain_two],
    verbose=True
)

product = "Queen Size Sheet Set"
overall_simple_chain.run(product)


#  输出
# {
# 	'Review': "Je trouve le goût médiocre. La mousse ne tient pas, c'est bizarre. J'achète les mêmes dans le commerce et le goût est bien meilleur...\nVieux lot ou contrefaçon !?",
# 	'English_Review': "I find the taste mediocre. The foam does not hold, it's strange. I buy the same ones in stores and the taste is much better...\nOld batch or counterfeit!?",
# 	'summary': "The reviewer finds the product's taste mediocre, the foam unstable, and suspects it might be an old batch or counterfeit compared to the superior versions bought in stores. \n\nHere's a one-sentence summary:\nThe reviewer deems the product's taste subpar and foam disappointing, questioning if it's an old batch or fake compared to the better-quality store-bought version.",
# 	'followup_message': 'La réponse de suivi pourrait être :\n\n"Nous sommes désolés de constater que vous avez eu une expérience décevante avec notre produit, notamment en ce qui concerne le goût et la stabilité de la mousse. Nous prenons vos inquiétudes au sujet d\'une possible ancienne loterie ou d\'un produit contrefait très au sérieux et investiguerons cette affaire de manière approfondie. Nous nous engageons à maintenir la qualité de nos produits et votre satisfaction est notre priorité."'
# }