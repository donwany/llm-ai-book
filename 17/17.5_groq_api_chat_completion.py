# Export the API key for Groq
# export GROQ_API_KEY=<your-api-key-here>

# Install the Groq package (uncomment to use)
# pip install groq

import os
from groq import Groq

# Initialize the Groq client using the API key from environment variables
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Create a chat completion request with the specified model
chat_completion = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": "Explain the importance of low latency LLMs"
    }],
    model="mixtral-8x7b-32768"
)

# Print the content of the response
print(chat_completion.choices[0].message.content)

# ------------------- OUTPUT -----------------------------
# {
#   "id": "34a9110d-c39d-423b-9ab9-9c748747b204",
#   "object": "chat.completion",
#   "created": 1708045122,
#   "model": "mixtral-8x7b-32768",
#   "system_fingerprint": null,
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "Low latency Large Language Models (LLMs) are important in the field of artificial intelligence and natural language processing (NLP) for several reasons:\n\n1. Real-time applications: Low latency ...."
#       },
#       "finish_reason": "stop",
#       "logprobs": null
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 24,
#     "completion_tokens": 377,
#     "total_tokens": 401,
#     "prompt_time": 0.009,
#     "completion_time": 0.774,
#     "total_time": 0.783
#   }
# }
