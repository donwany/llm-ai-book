#!/bin/bash

# Define model and volume
model="mistralai/Mistral-7B-Instruct-v0.1"
volume="$PWD/data"

# Run Docker container with the specified model and volume
docker run \
    --gpus all \
    --shm-size 1g \
    -p 8080:80 \
    -v "$volume:/data" \
    ghcr.io/predibase/lorax:latest \
    --model-id "$model"

# Use curl to send a POST request to the running container
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{
        "inputs": "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]",
        "parameters": {
            "max_new_tokens": 64
        }
    }' \
    -H 'Content-Type: application/json'
