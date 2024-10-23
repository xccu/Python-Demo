#根据输入的自然语言指令和图像信息完成任务，推荐使用 SSE 或同步调用方式请求接口

#https://open.bigmodel.cn/dev/api#glm-4

# 需要安装：
#   !pip install zhipuai      5.4.1
from zhipuai import ZhipuAI

from configure import get_yaml

key = get_yaml('zhipu.key')
client = ZhipuAI(api_key=key) # 填写您自己的APIKey
response = client.chat.completions.create(
    model="glm-4v",  # 填写需要调用的模型名称
    messages=[
       {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "图里有什么，人物是谁"
          },
          {
            "type": "image_url",
            "image_url": {
                "url" : "https://i0.hdslb.com/bfs/article/b295df950f3e2eb9f2202df819e26fe0dcd943ef.jpg"
            }
          }
        ]
      }
    ]
)
print(response.choices[0].message)
