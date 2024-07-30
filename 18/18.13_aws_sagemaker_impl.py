# pip install "sagemaker>=2.216.0" --upgrade --quiet

import sagemaker
import boto3
import json
from sagemaker.huggingface import HuggingFaceModel

# Initialize SageMaker session
sess = sagemaker.Session()
sagemaker_session_bucket = None

if sagemaker_session_bucket is None and sess is not None:
    sagemaker_session_bucket = sess.default_bucket()

# Get SageMaker execution role
try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

# Set up SageMaker session with default bucket
sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

# Print SageMaker role ARN and session region
print(f"Sagemaker role ARN: {role}")
print(f"Sagemaker session region: {sess.boto_region_name}")

# Define the Hugging Face model image URI
llm_image = f"763104351884.dkr.ecr.{sess.boto_region_name}.amazonaws.com/huggingface-pytorch-tgi-inference:2.1-tgi2.0-gpu-py310-cu121-ubuntu22.04"
print(f"LLM image URI: {llm_image}")

# SageMaker instance configuration
instance_type = "ml.p4d.24xlarge"
health_check_timeout = 900

# Define model and endpoint configuration parameters
config = {
    'HF_MODEL_ID': "meta-llama/Meta-Llama-3-8B-Instruct", # Model ID from Hugging Face Hub
    'SM_NUM_GPUS': "8", # Number of GPUs used per replica
    'MAX_INPUT_LENGTH': "2048", # Max length of input text
    'MAX_TOTAL_TOKENS': "4096", # Max length of the generation (including input text)
    'MAX_BATCH_TOTAL_TOKENS': "8192", # Limits the number of tokens that can be processed in parallel during generation
    'MESSAGES_API_ENABLED': "true", # Enable the messages API
    'HUGGING_FACE_HUB_TOKEN': "<REPLACE WITH YOUR TOKEN>" # Replace with your Hugging Face Hub token
}

# Check if Hugging Face Hub token is set
assert config['HUGGING_FACE_HUB_TOKEN'] != "<REPLACE WITH YOUR TOKEN>", "Please set your Hugging Face Hub token"

# Create HuggingFaceModel with the image URI
llm_model = HuggingFaceModel(role=role, image_uri=llm_image, env=config)

# Deploy to an endpoint
llm = llm_model.deploy(
    initial_instance_count=1,
    instance_type=instance_type,
    container_startup_health_check_timeout=health_check_timeout,
)

# Run inference and chat with the model
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is large language model?"}
]

parameters = {
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "top_p": 0.6,
    "temperature": 0.9,
    "max_tokens": 512,
    "stop": [""],
}

# Get the model response
chat = llm.predict({"messages": messages, **parameters})
print(chat["choices"][0]["message"]["content"].strip())

# Clean-up
llm.delete_model()
llm.delete_endpoint()
