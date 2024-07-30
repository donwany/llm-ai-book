# Install the Ollama library
# pip install ollama

import ollama

# Chat with a model and get a response
response = ollama.chat(
    model='llama2',
    messages=[{
        'role': 'user',
        'content': 'Why is the sky blue?',
    }]
)
print(response['message']['content'])

# Streaming response from the model
stream = ollama.chat(
    model='llama2',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    stream=True
)
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)

# Simple chat with the model (no output handling shown)
ollama.chat(
    model='llama2',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}]
)
