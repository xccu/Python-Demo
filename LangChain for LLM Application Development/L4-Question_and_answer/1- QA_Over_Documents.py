# pip install --upgrade langchain
# pip install docarray

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.llms import OpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.chat_models import ChatZhipuAI
from langchain_openai import OpenAIEmbeddings

from configure import get_yaml
from utils import load_env
from utils.jwt_token import generate_zhipu_token

#加载本地环境变量文件 .env

load_env.load()

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file,encoding='utf-8')
embeddings_model = OpenAIEmbeddings(model="embedding-2")


#_base_client.base_url
#note: need to config OPENAI_API_BASE in .env
#<Request('POST', 'https://api.openai.com/v1/embeddings')>
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."

# llm_replacement_model  = ChatOpenAI(
#     model_name= "glm-4",
#     openai_api_base= "https://open.bigmodel.cn/api/paas/v4",
#     openai_api_key=generate_zhipu_token()
# )

llm_replacement_model = ChatOpenAI(temperature = 0.0)
#llm_replacement_model = ChatZhipuAI(model="glm-3-turbo",temperature=0.0)


response = index.query(query, llm = llm_replacement_model)

display(Markdown(response))