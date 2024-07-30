#!/bin/bash

# Clone the LLaMA-Factory repository from GitHub
git clone https://github.com/hiyouga/LLaMA-Factory.git

# Change directory to the cloned repository
cd LLaMA-Factory

# Create a new Conda environment named 'llama_factory' with Python 3.11
conda create -n llama_factory python=3.11

# Activate the 'llama_factory' Conda environment
conda activate llama_factory

# Install the required Python packages from the requirements file
pip install -r requirements.txt

# Install additional packages
pip install bitsandbytes transformers_stream_generator

# Upgrade the huggingface_hub package to the latest version
pip install --upgrade huggingface_hub

# Log in to Hugging Face Hub with your authentication token
huggingface-cli login --token=<hf...>

# Start the LLaMA-Factory web UI
CUDA_VISIBLE_DEVICES=0 GRADIO_SHARE=1 llamafactory-cli webui

# Build the Docker image using the Dockerfile in the current directory
docker build -f ./Dockerfile -t llama-factory:latest .

# Run the Docker container with GPU support and volume mounts for caching, data, and output directories
docker run --gpus=all \
    -v ./hf_cache:/root/.cache/huggingface/ \  # Mount Hugging Face cache directory
    -v ./data:/app/data \                      # Mount data directory
    -v ./output:/app/output \                  # Mount output directory
    -e CUDA_VISIBLE_DEVICES=0 \                # Set environment variable for CUDA device
    -p 7860:7860 \                             # Expose port 7860 for web UI
    --shm-size 16G \                           # Set shared memory size to 16GB
    --name llama_factory \                     # Name the container 'llama_factory'
    -d llama-factory:latest                    # Run container in detached mode using the built image

# Alternatively, use Docker Compose to start the container based on the configuration in docker-compose.yml
docker compose -f ./docker-compose.yml up -d

# Upload the trained model to Hugging Face Hub
huggingface-cli upload <USERNAME>/MY-MODELNAME saves/Qwen-1.8B-Chat/lora/train_2024
