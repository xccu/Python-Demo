#!pip install DateTime

from langchain.agents import tool
from datetime import date
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
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

@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())

agent= initialize_agent(
    tools + [time],
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)