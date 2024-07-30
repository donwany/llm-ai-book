# Install the necessary packages (uncomment to use)
# pip install --upgrade langchain langchain-google-vertexai

# Import the necessary modules
from langchain_google_vertexai import (
    GemmaChatVertexAIModelGarden,
    GemmaVertexAIModelGarden,
)
from langchain_core.messages import HumanMessage

# Define parameters
endpoint_id = "YOUR_ENDPOINT_ID"
project = "YOUR_PROJECT"
location = "YOUR_LOCATION"

# Initialize the GemmaVertexAIModelGarden
llm = GemmaVertexAIModelGarden(
    endpoint_id=endpoint_id,
    project=project,
    location=location,
)

# Invoke the model with a query
output = llm.invoke("What is the meaning of life?")
print(output)

# Initialize the GemmaChatVertexAIModelGarden
llm_chat = GemmaChatVertexAIModelGarden(
    endpoint_id=endpoint_id,
    project=project,
    location=location,
)

# Create and send messages
message1 = HumanMessage(content="How much is 2+2?")
answer1 = llm_chat.invoke([message1])
print(answer1)

message2 = HumanMessage(content="How much is 3+3?")
answer2 = llm_chat.invoke([message1, answer1, message2])
print(answer2)

# Invoke the model with parsing responses
answer1 = llm_chat.invoke([message1], parse_response=True)
print(answer1)

answer2 = llm_chat.invoke([message1, answer1, message2], parse_response=True)
print(answer2)

# Running from Hugging Face
from langchain_google_vertexai import GemmaChatLocalHF, GemmaLocalHF

hf_access_token = "PUT_YOUR_TOKEN_HERE"
model_name = "google/gemma-2b"

# Initialize GemmaLocalHF
llm_hf = GemmaLocalHF(model_name=model_name, hf_access_token=hf_access_token)
output_hf = llm_hf.invoke("What is the meaning of life?", max_tokens=50)
print(output_hf)
