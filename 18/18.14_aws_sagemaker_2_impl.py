# pip install "sagemaker>=2.216.0" --upgrade --quiet
# pip install transformers

import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel
from transformers import AutoTokenizer
import json

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

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=config['HUGGING_FACE_HUB_TOKEN'])

# Prompt to generate
messages = [
    {"role": "system", "content": "You are a friendly AI engineer answering AI questions."},
    {"role": "user", "content": "How is large language model?"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("")
]

# Generation arguments
payload = {
    "max_new_tokens": 2048,
    "eos_token_id": terminators,
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.9,
    "return_full_text": False,
}

# Run inference
response = llm.predict({"inputs": prompt, "parameters": payload})
print(response[0]['generated_text'])

# Clean-up
llm.delete_model()
llm.delete_endpoint()
