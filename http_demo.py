import requests
import json

from configure import get_yaml
from utils.jwt_token import generate_token

key = get_yaml('zhipu.key')
exp = get_yaml('zhipu.exp')

token = generate_token(key,exp)
authorization = 'Bearer {}'.format(token)

headers = {
    'Authorization': authorization,
    'Content-Type': 'application/json; charset=UTF-8',
    #"Accept": "application/json",
    #"ZhipuAI-SDK-Ver": "v2.0.1",
    #"source_type": "zhipu-sdk-python",
    #"x-request-sdk": "zhipu-sdk-python",
}

data = {
	"model": "glm-4",
	"messages": [{
		"role": "user",
		"content": "智普支持langchain调用吗"
	}]
}

json_str = json.dumps(data)

response = requests.post('https://open.bigmodel.cn/api/paas/v4/chat/completions',data=json_str,headers=headers)
print(response.text)