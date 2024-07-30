# Clone the llama.cpp repository from GitHub
# git clone https://github.com/ggerganov/llama.cpp
# Change directory to llama.cpp, set LLAMA_CUBLAS flag to 1, and compile the code
# cd llama.cpp && LLAMA_CUBLAS=1 make
# Install the required dependencies for the conversion script
# pip install -r requirements/requirements-convert-hf-to-gguf.txt

import os
from huggingface_hub import snapshot_download

# Define the model name to be downloaded from Hugging Face
model_name = "Qwen/Qwen1.5-1.8B"
# Define the quantization methods to be used
methods = ['q4_k_m']
# Set the paths for the base model and quantized model
base_model = "./original_model/"
quantized_path = "./quantized_model/"

# Download the model from Hugging Face Hub to the specified local directory
snapshot_download(repo_id=model_name, local_dir=base_model, local_dir_use_symlinks=False)

# Create the directory for the quantized models
os.makedirs(quantized_path, exist_ok=True)

# Convert the Hugging Face model to GGUF format with FP16 precision
os.system("python llama.cpp/convert-hf-to-gguf.py ./original_model/ --outtype f16 --outfile ./quantized_model/FP16.gguf")

# Loop through the defined quantization methods and quantize the model accordingly
for m in methods:
    qtype = f"{quantized_path}/{m.upper()}.gguf"
    os.system(f"./llama.cpp/quantize {quantized_path}/FP16.gguf {qtype} {m}")


# Run the main llama.cpp script with the quantized model and specified parameters
os.system("./llama.cpp/main -m ./quantized_model/Q4_K_M.gguf \
    -n 90 \
    --repeat_penalty 1.0 \
    --color -i -r \"User:\" \
     -f llama.cpp/prompts/chat-with-bob.txt")

# Upload the Quantized Model to Hugging Face Hub:
from huggingface_hub import HfApi, create_repo

# Define the path to the quantized model
model_path = "./quantized_model/Q4_K_M.gguf"
# Define the name for the new repository on Hugging Face Hub
repo_name = "qwen1.5-llm"
# Create a new repository on Hugging Face Hub (set to public)
repo_url = create_repo(repo_name, private=False)

# Initialize the HfApi object for interacting with the Hugging Face Hub
api = HfApi()
# Upload the quantized model file to the created repository
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="Q4_K_M.gguf",
    repo_id=f"worldboss/{repo_name}",
    repo_type="model",
)
