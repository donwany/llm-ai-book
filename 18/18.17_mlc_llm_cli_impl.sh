#!/bin/bash

# Activate your conda environment
conda activate your-environment

# Install mlc-llm-nightly and mlc-ai-nightly packages
python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly mlc-ai-nightly

# Check available commands for mlc_llm chat
mlc_llm chat -h

# Use prebuilt model with mlc_llm
mlc_llm chat HF://mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC \
--device "cuda:0" \
--overrides context_window_size=1024

# Use locally compiled weights with mlc_llm
mlc_llm chat dist/Llama-2-7b-chat-hf-q4f16_1-MLC \
--device "cuda:0" --overrides context_window_size=1024 \
--model-lib-path dist/prebuilt_libs/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-vulkan.so
