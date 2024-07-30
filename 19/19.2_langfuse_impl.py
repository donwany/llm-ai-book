# Install langfuse
# pip install langfuse

# Clone the langfuse repository
# git clone https://github.com/langfuse/langfuse.git
# cd langfuse
# Run the server and database using Docker
# docker compose up -d

import os

# Set environment variables
os.environ["LANGFUSE_PUBLIC_KEY"] = "<your_langfuse_public_key>"
os.environ["LANGFUSE_SECRET_KEY"] = "<your_langfuse_secret_key>"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"
os.environ["OPENAI_API_KEY"] = "<your_openai_api_key>"

# Use Langfuse with OpenAI Integration
from langfuse.decorators import observe
from langfuse.openai import openai  # OpenAI integration


@observe()
def story():
    return openai.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=100,
        messages=[
            {"role": "system", "content": "You are a great storyteller."},
            {"role": "user", "content": "Once upon a time in a galaxy far, far away..."}
        ],
    ).choices[0].message.content


@observe()
def main():
    return story()


# Run the main function to generate a story
main()
