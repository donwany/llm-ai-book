# Install the necessary libraries
# !pip install anthropic
# !pip install -U anthropic[bedrock]
# !pip install -U anthropic[vertex]

import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from anthropic import AnthropicBedrock, AnthropicVertex

# Initialize Anthropic client with API key
api_key = os.environ.get("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key)

# Generate a message using Claude model
response = client.messages.create(
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello, Claude"}],
    model="claude-3-opus-20240229"
)
print("Claude Response:", response.content)

# ----------------------- Text Generation with Anthropic -------------------
# Generate completion using Claude 2.1 model
completion = client.completions.create(
    model="claude-2.1",
    max_tokens_to_sample=1024,
    prompt=f"{HUMAN_PROMPT} Hello, Claude{AI_PROMPT}"
)
print("Claude 2.1 Completion:", completion.content)

# ------------------------ Amazon Bedrock API -------------------------------
# Initialize Bedrock client with AWS credentials
bedrock_client = AnthropicBedrock(
    aws_access_key="<access key>",
    aws_secret_key="<secret key>",
    aws_session_token="<session_token>",
    aws_region="us-west-2"
)

# Generate a message using Claude model on Bedrock
bedrock_response = bedrock_client.messages.create(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello, world"}]
)
print("Bedrock Response:", bedrock_response.content)

# --------------------- VertexAI via Anthropic ---------------------------
# Initialize VertexAI client with project and region
project_id = "MY_PROJECT_ID"
region = "MY_REGION"
vertex_client = AnthropicVertex(
    project_id=project_id,
    region=region
)

# Generate a message using Claude model on VertexAI
vertex_response = vertex_client.messages.create(
    model="claude-3-haiku@20240307",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hey Claude!"}]
)
print("VertexAI Response:", vertex_response.content)
