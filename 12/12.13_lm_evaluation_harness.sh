#!/bin/bash

# --------- Setup ---------
# Clone the repository
# git clone https://github.com/EleutherAI/lm-evaluation-harness
# Change directory to lm-evaluation-harness
# cd lm-evaluation-harness
# Install the package
# pip install -e .

# --------- Evaluate on Hellaswag ---------
# Evaluate the model on the Hellaswag dataset
lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8

# --------- Evaluate on Hellaswag and Lambada ---------
# Evaluate the model on both Hellaswag and Lambada datasets with specified arguments
lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B,revision=step100000,dtype="float" \
    --tasks lambada_openai,hellaswag \
    --device cuda:0 \
    --batch_size 8

# Evaluate with automatic batch size adjustment
lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B,revision=step100000,dtype="float" \
    --tasks lambada_openai,hellaswag \
    --device cuda:0 \
    --batch_size auto:4

# --------- Evaluate with OpenAI API ---------
# Set the OpenAI API key as an environment variable
export OPENAI_API_KEY=YOUR_KEY_HERE
# Evaluate the model using OpenAI's completions API
lm_eval --model openai-completions \
    --model_args model=davinci \
    --tasks lambada_openai,hellaswag

# --------- Evaluate with Advanced Settings ---------
# Evaluate the model on multiple tasks with advanced settings
lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B,parallelize=True,load_in_4bit=True,peft=nomic-ai/gpt4all-j-lora \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --device cuda:0
