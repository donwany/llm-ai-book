# Clone the repository and install dependencies
# git clone -b mixtral https://github.com/arcee-ai/mergekit.git
# cd mergekit && pip install -e .
# pip install --upgrade transformers
# pip install bitsandbytes accelerate

from huggingface_hub import notebook_login, HfApi
import transformers
from transformers import AutoTokenizer
import os
import torch

# Log in to the Hugging Face account from the notebook
notebook_login()

# Define the configuration for merging models
merge_config = """
base_model: NousResearch/Hermes-2-Pro-Mistral-7B
dtype: float16
gate_mode: cheap_embed
experts:
  - source_model: BioMistral/BioMistral-7B-DARE
    positive_prompts: "As a doctor of medicine,"
  - source_model: PocketDoc/Dans-AdventurousWinds-7b
    positive_prompts: "[Genres: Science Fiction]\n[Tags: humor, old school, sci fi]"
"""

# Write the merge configuration to a YAML file
with open('config.yaml', 'w') as f:
    f.write(merge_config)

######################
# USING MERGEKIT
######################
# Execute the merge using the mergekit CLI
os.system("mergekit-moe config.yaml merge --copy-tokenizer --allow-crimes --out-shard-size 1B --trust-remote-code")

# Set the username and model name for the new merged model
USERNAME = "worldboss"
MODEL_NAME = "MedTral-2x7b"

# Initialize the Hugging Face API with a token
api = HfApi(token="hf_cx...")  # Replace with your actual token

# Create a new repository for the merged model on Hugging Face
api.create_repo(repo_id=f"{USERNAME}/{MODEL_NAME}", repo_type="model")

# Upload the contents of the 'merge' folder to the newly created repository
api.upload_folder(repo_id=f"{USERNAME}/{MODEL_NAME}", folder_path="merge")

# Specify the new model ID after merging
new_model = "worldboss/MedTral-4x7b"

# Initialize the tokenizer with the new merged model
tokenizer = AutoTokenizer.from_pretrained(new_model)

# Create a text generation pipeline with the specified model and settings
pipeline = transformers.pipeline(
    "text-generation",
    model=new_model,
    model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True}  # Model settings for inference
)

# Define the chat messages with roles and contents
messages = [{"role": "user", "content": "Do you know how to write a Medical Claims?"}]

# Apply a chat template to the messages without tokenizing and add a generation prompt
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Generate text based on the prompt with specified parameters
# max_new_tokens=256: Generate up to 256 new tokens
# do_sample=True: Use sampling instead of greedy decoding
# temperature=0.7: Control randomness in sampling (lower is less random)
# top_k=50: Consider only the top 50 tokens for sampling
# top_p=0.95: Use nucleus sampling to consider tokens that make up 95% of the probability mass
outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

# Print the generated text from the first output
print(outputs[0]["generated_text"])
