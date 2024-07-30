# Install requests
# python -m pip install requests

import requests

url = "https://..."

payload = {
    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "prompt": "<s>[INST] What is the capital of Ghana? [/INST]",
    "max_tokens": 512,
    "stop": ["</s>", "[/INST]"],
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "repetition_penalty": 1,
    "n": 1
}

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": "Bearer f65d192116989998f3cd49b7a2f9f8a7a2e483d9527894a877a16af17a8fb178"
}

response = requests.post(url, json=payload, headers=headers)
print(response.text)
