#!/bin/bash

# brew install llama.cpp

# Using Docker
sudo docker run \
        --gpus all \
        -v /path/to/models:/models local/llama.cpp:light-cuda \
        -m /models/7B/ggml-model-q4_0.gguf \
        -p "what is machine learning?:" \
        -n 512 \
        --n-gpu-layers 99

# Using CLI
llama-server \
        --hf-repo TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
        --hf-file mistral-7b-instruct-v0.2.Q5_K_M.gguf

# Inference
curl --request POST \
        --url http://localhost:8080/completion \
        --header "Content-Type: application/json" \
        --data '{"prompt": "what is machine learning?","n_predict": 128}'