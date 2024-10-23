from zhipuai import ZhipuAI

from configure import get_yaml

key = get_yaml('zhipu.key')

client = ZhipuAI(api_key=key)
response = client.embeddings.create(
    model="embedding-2", #填写需要调用的模型名称
    input="你好",
)
client.files.create()
print(response)
