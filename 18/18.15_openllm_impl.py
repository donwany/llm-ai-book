# pip install openllm "openllm[llama]" "openllm[mistral]"
# Start LLM servers
# TRUST_REMOTE_CODE=True openllm start microsoft/phi-2
# TRUST_REMOTE_CODE=True openllm start mistralai/Mistral-7B-Instruct-v0.1
# openllm start meta-llama/Llama-2-7b-chat-hf --backend vllm

# Open in the browser
# Open http://0.0.0.0:3000

import openllm

# Create an HTTP client for OpenLLM server
client = openllm.client.HTTPClient('http://localhost:3000')

# Query the model
response = client.query('Explain to me the difference between "llm" and "lmm"')
print(response)

# using quantized model
# Start models with quantization
# openllm start TheBloke/Llama-2-7B-Chat-GPTQ --quantize gptq
# openllm start microsoft/phi-2 --quantize int8
# openllm start TheBloke/zephyr-7B-alpha-AWQ --quantize awq

# using bentoml to deploy models
import bentoml

# Initialize the LLM
llm = openllm.LLM('microsoft/phi-2')

# Create a BentoML service
svc = bentoml.Service(name='llm-phi-service', runners=[llm.runner])

@svc.api(input=bentoml.io.Text(), output=bentoml.io.Text())
async def prompt(input_text: str) -> str:
    generation = await llm.generate(input_text)
    return generation.outputs[0].text

# using langchain with openllm
from langchain.llms import OpenLLM

# Initialize the LLM with LangChain
llm = OpenLLM(model_name='llama', model_id='meta-llama/Llama-2-7b-hf')

# Query the model
response = llm('What is the difference between a duck and a goose? And why are there so many geese in Canada?')
print(response)


# running openllm with docker
# docker run --rm -it -p 3000:3000 ghcr.io/bentoml/openllm start facebook/opt-1.3b --backend pt
# docker run --rm --gpus all -p 3000:3000 -it ghcr.io/bentoml/openllm start HuggingFaceH4/zephyr-7b-beta --backend vllm



