#!/bin/bash

# Create a new Conda environment named 'mlx' with Python 3.11
conda create -n mlx python=3.11

# Activate the 'mlx' Conda environment
conda activate mlx

# Clone the MLX examples repository from GitHub
git clone https://github.com/ml-explore/mlx-examples

# Navigate into the cloned directory
cd mlx-examples/

# Install required Python packages from the requirements file
pip install -r requirements.txt

# Install additional MLX packages and utilities
pip install mlx mlx-llm huggingface_hub hf_transfer

# ------------------ Python API -----------------------------
# Import necessary functions from the MLX LLM module
from mlx_lm import load, generate, convert

# Load the pre-trained Mistral-7B-Instruct model and tokenizer
model, tokenizer = load("mistralai/Mistral-7B-Instruct-v0.1")

# Generate a response using the loaded model with a given prompt
response = generate(model, tokenizer, prompt="hello", verbose=True)

# ------------------ Convert model using Python API ---------
# Define the upload repository for the quantized model
upload_repo = "mlx-community/My-Mistral-7B-v0.1-4bit"

# Convert the Mistral-7B-Instruct model to a quantized format and upload to the specified repository
convert("mistralai/Mistral-7B-v0.1", quantize=True, upload_repo=upload_repo)

# ------------------ CLI generate ---------------------------
# Generate a response using the CLI with the Mistral-7B-Instruct model
mlx_lm.generate --model mistralai/Mistral-7B-Instruct-v0.1 --prompt "hello"

# ----------------- Quantize model from CLI -----------------
# Quantize the Mistral-7B-Instruct model using the CLI
mlx_lm.convert --hf-path mistralai/Mistral-7B-Instruct-v0.1 -q

# ----------------- Upload model to Hugging Face -------------
# Quantize the Mistral-7B model and upload to the Hugging Face repository
mlx_lm.convert --hf-path mistralai/Mistral-7B-v0.1 -q --upload-repo mlx-community/my-4bit-mistral

# -------------------- Download model -----------------------
# Enable Hugging Face transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download a model from Hugging Face to a local directory
huggingface-cli download --local-dir Mixtral-8x7B-Instruct-v0.1 mlx-community/Mixtral-8x7B-Instruct-v0.1

# Run an example script with the downloaded model
python llm/mixtral/mixtral.py --model_path Mixtral-8x7B-Instruct-v0.1 --prompt "What is large language model"

# -------------------- Fine-tune model ----------------------
# Fine-tune the Mistral-7B model with LoRA using the provided script
python llm/lora/lora.py --model mistralai/Mistral-7B-v0.1 \
   --train --batch-size 1 --lora-layers 4

# Merge the base model and the uploaded model with the provided script
python llm/lora/fuse.py --upload-name <USERNAME>/<MODEL_NAME> --hf-repo mistralai/Mistral-7B-v0.1

# -------------------- Stable Diffusion ---------------------
# Generate images from a text prompt using Stable Diffusion
python stable_diffusion/txt2image.py "A photo of an astronaut riding a horse on Mars." --n_images 4 --n_rows 2

# Modify an existing image using a new prompt
python stable_diffusion/image2image.py --strength 0.5 original.png 'A lit fireplace'
