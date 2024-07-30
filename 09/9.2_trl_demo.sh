#!/bin/bash

# pip install trl
# pip install git+https://github.com/huggingface/trl.git

# SFT CLI
trl sft \
    --model_name_or_path facebook/opt-125m \
    --dataset_name imdb \
    --output_dir opt-sft-imdb

# DPO CLI
trl dpo \
    --model_name_or_path facebook/opt-125m \
    --dataset_name trl-internal-testing/hh-rlhf-trl-style \
    --output_dir opt-sft-hh-rlhf

# Chat CLI
trl chat --model_name_or_path Qwen/Qwen1.5-0.5B-Chat