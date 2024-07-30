# export GROQ_API_KEY=<your-api-key-here>
# pip install groq
from groq import Groq

client = Groq()
completion = client.chat.completions.create(
    model="mixtral-8x7b-32768",
    messages=[{"role": "user", "content": ""}],
    temperature=0.5, max_tokens=1024,
    top_p=1, stream=True, stop=None,
)
for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")