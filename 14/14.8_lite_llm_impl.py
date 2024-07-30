# Install the necessary library and clone the repository
# pip install litellm
# git clone https://github.com/BerriAI/litellm.git

import os
from litellm import completion

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["COHERE_API_KEY"] = "your-cohere-key"
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"
os.environ["HUGGINGFACE_API_KEY"] = "huggingface_api_key"

# Define the messages for the API calls
messages = [{"content": "Hello, how are you?", "role": "user"}]


# ------------------- OpenAI API Call ----------------------------
def call_openai(messages):
    response = completion(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response


# ------------------- Cohere API Call -----------------------------
def call_cohere(messages):
    response = completion(
        model="command-nightly",
        messages=messages
    )
    return response


# ------------------- Anthropic API Call --------------------------
def call_anthropic(messages):
    response = completion(
        model="claude-2",
        messages=messages
    )
    return response


# ------------------- Hugging Face API Call ------------------------
def call_huggingface(messages):
    response = completion(
        model="huggingface/WizardLM/WizardCoder-Python-34B-V1.0",
        messages=messages,
        api_base="https://my-endpoint.huggingface.cloud"
    )
    return response


# ------------------ Streaming Response -----------------------
def call_openai_streaming(messages):
    response = completion(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True
    )
    return response


# Make API calls and print responses
openai_response = call_openai(messages)
cohere_response = call_cohere(messages)
anthropic_response = call_anthropic(messages)
huggingface_response = call_huggingface(messages)
streaming_response = call_openai_streaming(messages)

# Print responses
print("OpenAI Response:", openai_response)
print("Cohere Response:", cohere_response)
print("Anthropic Response:", anthropic_response)
print("Hugging Face Response:", huggingface_response)
print("Streaming Response:", streaming_response)
