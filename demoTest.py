from zhipuai import ZhipuAI
from configure import get_yaml

key = get_yaml('zhipu.key')

# 同步调用
client = ZhipuAI(api_key=key) # 填写您自己的APIKey
response = client.chat.completions.create(
    model="glm-4",  # 填写需要调用的模型名称
    messages=[
        {"role": "user", "content": "请回答问题"},
        {"role": "user", "content": "智谱AI调用SDK需要联网吗"},
    ],
)
print(response.choices[0].message)
