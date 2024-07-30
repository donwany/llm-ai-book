#!/bin/bash

# pip install skypilot
sky check

# create serving.yaml file

#resources:
#    accelerators: A100
#envs:
#    MODEL_NAME: decapoda-research/llama-13b-hf
#    TOKENIZER: hf-internal-testing/llama-tokenizer
#setup: |
#    conda create -n vllm python=3.9 -y && conda activate vllm
#    git clone https://github.com/vllm-project/vllm.git
#    cd vllm
#    pip install .
#    pip install gradio
#run: |
#    conda activate vllm
#    echo 'Starting vllm api server...'
#    python -u -m vllm.entrypoints.api_server \
#        --model $MODEL_NAME \
#        --tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE \
#        --tokenizer $TOKENIZER 2>&1 | tee api_server.log &
#    echo 'Waiting for vllm api server to start...'
#    while ! `cat api_server.log | grep -q 'Uvicorn running on'`; do sleep 1; done
#    echo 'Starting gradio server...'
#    python vllm/examples/gradio_webserver.py

sky launch serving.yaml