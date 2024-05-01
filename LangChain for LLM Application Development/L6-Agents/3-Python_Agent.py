#!pip install -U wikipedia

from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI

# read local .env file
# load_env.load()


#llm = ChatOpenAI(temperature=0, model=llm_model)
from utils.jwt_token import generate_zhipu_token

llm = ChatOpenAI(
    model_name= "glm-4",
    openai_api_base= "https://open.bigmodel.cn/api/paas/v4",
    openai_api_key=generate_zhipu_token()
)

tools = load_tools(["llm-math","wikipedia"], llm=llm)


#Python Agent
agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
)

customer_list = [["Harrison", "Chase"],
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"],
                 ["Geoff","Fusion"],
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]

agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""")