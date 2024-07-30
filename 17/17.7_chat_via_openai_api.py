from openai import OpenAI

# Initialize the OpenAI client with the API key and base URL
client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8080/v1")

# Create a chat completion request with a specific model and messages
resp = client.chat.completions.create(
    model="alignment-handbook/zephyr-7b-dpo-lora",
    messages=[
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "How many helicopters can a human eat in one sitting?",
        },
    ],
    max_tokens=100,
)

# Print the content of the response
print("Response:", resp.choices[0].message.content)
