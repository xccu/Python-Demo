#!pip install -U wikipedia

from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
#from langchain.tools.python.tool import PythonREPLTool
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

agent= initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)

agent("What is the 25% of 300?")