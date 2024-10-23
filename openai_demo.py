#pip install openai
import openai
from openai import OpenAI

# 加载本地 .env 文件
from utils import load_env

load_env.load()

client = OpenAI()
openai.api_timeout = 600
messages = []
system_message = input("What type of chatbot you want me to be?")
messages.append({"role":"system","content":system_message})
print("Alright! I am ready to be your friendly chatbot" + "\n" + "You can now type your messages.")

message = input("")
messages.append({"role":"user","content": message})
response=client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=messages
)

print(response.choices[0].message)