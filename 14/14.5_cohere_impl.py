# pip install cohere
import cohere

co = cohere.Client(api_key="YOUR_API_KEY", )
chat = co.chat(message="hello world!", model="command")
print(chat)

# Streaming
co = cohere.Client(api_key="YOUR_API_KEY", )
stream = co.chat_stream(message="Tell me a short story")
for event in stream:
    if event.event_type == "text-generation":
        print(event.text, end='')
