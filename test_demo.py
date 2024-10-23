from zhipuai import ZhipuAI
from configure import get_yaml

key = get_yaml('zhipu.key')

# 同步调用
client = ZhipuAI(api_key=key) # 填写您自己的APIKey
response = client.chat.completions.create(
    model="glm-4",  # 填写需要调用的模型名称
    messages=[
        {"role": "user", "content": "请回答问题"},
        {"role": "user", "content": "智谱AI如何兼容使用Langchain的VectorstoreIndexCreator函数，请给出示例代码"},
    ],
)
print(response.choices[0].message)