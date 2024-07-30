#!/bin/bash

git clone https://github.com/huggingface/text-generation-inference.git
cd text-generation-inference
make install

text-generation-launcher --help

# running locally
text-generation-launcher --model-id mistralai/Mistral-7B-Instruct-v0.2

# quantization
text-generation-launcher \
    --model-id mistralai/Mistral-7B-Instruct-v0.2 \
    --quantize bitsandbytes-nf4

# using docker
model=meta-llama/Llama-2-7b-chat-hf
volume=$PWD/data # Share a volume with the Docker container to avoid downloading weights every run
token=<your cli READ token>

docker run \
    --gpus all \
    --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token \
    -p 8080:80 \
    -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.4 \
    --model-id $model

# Inference
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
