
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from utils.jwt_token import generate_zhipu_token
from utils import load_env

# read local .env file
load_env.load()

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file,encoding='utf-8')
data = loader.load()

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

#llm = ChatOpenAI(temperature = 0.0, model=llm_model)
llm = ChatOpenAI(
    model_name= "glm-4",
    openai_api_base= "https://open.bigmodel.cn/api/paas/v4",
    openai_api_key=generate_zhipu_token()
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)
# Coming up with test datapoints
data[10]
data[11]

# Hard-coded examples
examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]

# LLM-Generated examples
from langchain.evaluation.qa import QAGenerateChain

#example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model))
example_gen_chain = QAGenerateChain.from_llm(llm)

# the warning below can be safely ignored
new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)
new_examples[0]
data[0]

#Combine examples
examples += new_examples
qa.run(examples[0]["query"])