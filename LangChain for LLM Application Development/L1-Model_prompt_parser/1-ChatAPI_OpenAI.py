#!pip install python-dotenv
#!pip install openai

from zhipuai import ZhipuAI
from configure import get_yaml

# style = """American English \
# in a calm and respectful tone
# """
#
# customer_email = """
# Arrr, I be fuming that me blender lid \
# flew off and splattered me kitchen walls \
# with smoothie! And to make matters worse,\
# the warranty don't cover the cost of \
# cleaning up me kitchen. I need yer help \
# right now, matey!
# """
#
# prompt = f"""Translate the text \
# that is delimited by triple backticks
# into a style that is {style}.
# text: ```{customer_email}```
# """

# Let's start with a direct API calls to OpenAI.
def get_completion(prompt):
    messages = [{"role": "user", "content": prompt}]

    key = get_yaml('zhipu.key')
    client = ZhipuAI(api_key=key)  # 填写您自己的APIKey
    # response = openai.ChatCompletion.create(
    response = client.chat.completions.create(
        model="glm-4",
        messages=messages,
        #temperature=0
    )
    return response.choices[0].message

result = get_completion("What is 1+1?")
print(result)

