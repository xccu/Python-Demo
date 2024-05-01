from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.embeddings import OpenAIEmbeddings

from utils import load_env
from utils.jwt_token import generate_zhipu_token



load_env.load()

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file,encoding='utf-8')

docs = loader.load()
docs[0]

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])


embeddings = OpenAIEmbeddings()

embed = embeddings.embed_query("Hi my name is Harrison")
print(len(embed))
print(embed[:5])

db = DocArrayInMemorySearch.from_documents(
    docs,
    embeddings
)

query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(query)

len(docs)
display(docs[0])

retriever = db.as_retriever()

#llm = ChatOpenAI(temperature = 0.0, model=llm_model)
llm = ChatOpenAI(
    model_name= "glm-4",
    openai_api_base= "https://open.bigmodel.cn/api/paas/v4",
    openai_api_key=generate_zhipu_token()
)

qdocs = "".join([docs[i].page_content for i in range(len(docs))])

response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.")

display(Markdown(response))

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)

query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."

response = qa_stuff.run(query)
display(Markdown(response))
response = index.query(query, llm=llm)

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])