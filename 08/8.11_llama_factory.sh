#!/bin/bash

git clone https://github.com/hiyouga/LLaMA-Factory.git
conda create -n llama_factory python=3.10
conda activate llama_factory
cd LLaMA-Factory
pip install -e .[metrics]

# local environment
export CUDA_VISIBLE_DEVICES=0 # set CUDA_VISIBLE_DEVICES=0 for Windows
python src/webui.py

# Using Docker
docker build -f ./Dockerfile -t llama-factory:latest .

docker run --gpus=all \
        -v ./hf_cache:/root/.cache/huggingface/ \
        -v ./data:/app/data \
        -v ./output:/app/output \
        -e CUDA_VISIBLE_DEVICES=0 \
        -p 7860:7860 \
        --shm-size 16G \
        --name llama_factory \
        -d llama-factory:latest

# Using Docker-Compose
docker compose -f ./docker-compose.yml up -d