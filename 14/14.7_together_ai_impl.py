# Install the necessary library
# pip install --upgrade together

import os
from openai import OpenAI
import together

# Set up OpenAI client with API key and base URL
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
client = OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url='https://api.together.xyz/v1'
)

# ---------------------- Chat Completion ------------------------------
# Create a chat completion request using the OpenAI client
chat_response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are an expert travel guide."},
        {"role": "user", "content": "Tell me fun things to do in Dallas, TX."}
    ],
    model="mistralai/Mixtral-8x7B-Instruct-v0.1"
)

# Print the response content from the chat completion
print("Chat Completion Response:", chat_response.choices[0].message.content)

# ---------------------- Query Model ------------------------------
# Set up Together API key
together.api_key = "xxxxx"  # Replace with your Together API Key

# Create a completion request using the Together library
completion_response = together.Complete.create(
    prompt="[INST] Tell me some fun things to do in NYC [/INST]",
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    max_tokens=500,
    temperature=0.8
)

# Extract and print the generated text from the completion response
generated_text = completion_response['output']['choices'][0]['text']
print("Together API Response:", generated_text)
