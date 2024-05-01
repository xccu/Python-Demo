# Hard-coded examples
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import ChatOpenAI

from utils import load_env
from utils.jwt_token import generate_zhipu_token

# read local .env file
load_env.load()

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

#llm = ChatOpenAI(temperature=0, model=llm_model)
llm = ChatOpenAI(
    model_name= "glm-4",
    openai_api_base= "https://open.bigmodel.cn/api/paas/v4",
    openai_api_key=generate_zhipu_token()
)

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)

# LLM assisted evaluation
predictions = qa.apply(examples)

from langchain.evaluation.qa import QAEvalChain

eval_chain = QAEvalChain.from_llm(llm)

graded_outputs = eval_chain.evaluate(examples, predictions)
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()

graded_outputs[0]