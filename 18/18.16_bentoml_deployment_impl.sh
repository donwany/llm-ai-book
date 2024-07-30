#!/bin/bash

# Clone the repository
git clone https://github.com/bentoml/BentoVLLM.git
cd BentoVLLM/mistral-7b-instruct

# Install dependencies
pip install -r requirements.txt && pip install -f -U "pydantic>=2.0"

# Start the BentoML service
bentoml serve .

# The server is now active at http://localhost:3000
# Output:
# 2024-01-18T07:51:30+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:VLLM" listening on http://localhost:3000 (Press CTRL+C to quit)

# access the model using python client
import bentoml

# Connect to the BentoML server
with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    # Generate a response from the model
    response_generator = client.generate(
        prompt="Explain superconductors to a five years old",
        tokens=None
    )

    # Print the response
    for response in response_generator:
        print(response)

# accessing the model using curl
# Make a request to the BentoML server using curl
curl -X 'POST' \
  'http://localhost:3000/generate' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{ "prompt": "Explain superconductors to a five years old", "tokens": null }'


# Deploy the service to BentoCloud
bentoml deploy .

# access the model in the cloud
import bentoml
# Connect to the cloud BentoML server
with bentoml.SyncHTTPClient("https://vllm-llama-7b-e3c1c7db.mt-guc1.bentoml.ai/generate") as client:
    # Generate a response from the model in the cloud
    result = client.generate(
        max_tokens=1024,
        prompt="Explain superconductors to a five years old"
    )

    # Print the result
    print(result)