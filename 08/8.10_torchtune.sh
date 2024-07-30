#!/bin/bash

git clone https://github.com/pytorch/torchtune.git
cd torchtune
pip install torchtune

tune --help
# download model from huggingface
tune download meta-llama/Llama-2-7b-hf \
        --output-dir /tmp/Llama-2-7b-hf \
        --hf-token <HF_TOKEN>

# Running fine-tuning recipes: Llama2 7B + LoRA on single GPU:
# https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_lora_single_device.yaml
tune run lora_finetune_single_device \
    --config llama2/7B_lora_single_device

# distributed training
# https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_full.yaml
tune run full_finetune_distributed \
    --nproc_per_node 2 \
    --config llama2/7B_full

# modify configurations
tune run lora_finetune_single_device \
        --config llama2/7B_lora_single_device \
        batch_size=8 \
        enable_activation_checkpointing=True \
        max_steps_per_epoch=128