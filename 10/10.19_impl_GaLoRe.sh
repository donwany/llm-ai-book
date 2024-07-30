#!/bin/bash

# Install the GaLore library for optimized training
pip install galore-torch
# Clone the GaLore repository from GitHub
git clone git@github.com:jiaweizzhao/GaLore.git
# Navigate into the GaLore directory
cd GaLore
# Install GaLore in editable mode
pip install -e .

# Import GaLore optimizers for memory-efficient training
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor

# Define parameter groups for the optimizer
# 'non_galore_params': Parameters not optimized with GaLore
# 'galore_params': Parameters optimized with GaLore, with additional settings
param_groups = [
    {'params': non_galore_params},  # Parameters not using GaLore optimizers
    {'params': galore_params,       # Parameters using GaLore optimizers
     'rank': 128,                   # Rank for the low-rank approximation
     'update_proj_gap': 200,        # Gap for updating projection weights
     'scale': 0.25,                 # Scaling factor for GaLore
     'proj_type': 'std'}            # Type of projection ('std' for standard)
]

# Initialize the GaLore AdamW optimizer with specified parameter groups
optimizer = GaLoreAdamW(param_groups, lr=0.01)  # Learning rate set to 0.01

# Command to train a 7B model on a single GPU with 24GB memory using GaLore
# This setup uses LLaMA-7B with 8-bit GaLore-Adam optimizer, single GPU, and activation checkpointing
# Adjust batch size and other training parameters according to GPU memory
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_7b.json \        # Path to the model configuration file
    --lr 0.005 \                                 # Learning rate for training
    --galore_scale 0.25 \                        # Scaling factor for GaLore
    --rank 1024 \                                # Rank for low-rank approximation
    --update_proj_gap 500 \                      # Gap for updating projection weights
    --batch_size 16 \                            # Batch size per GPU
    --total_batch_size 512 \                     # Total batch size (effective with gradient accumulation)
    --activation_checkpointing \                 # Enable activation checkpointing to save memory
    --num_training_steps 150000 \                # Total number of training steps
    --warmup_steps 15000 \                       # Number of warmup steps
    --weight_decay 0 \                           # Weight decay for regularization (set to 0)
    --grad_clipping 1.0 \                        # Gradient clipping value
    --dtype bfloat16 \                           # Data type for computations (bfloat16 for efficiency)
    --eval_every 1000 \                          # Evaluation frequency (every 1000 steps)
    --single_gpu \                               # Specify that training is done on a single GPU
    --optimizer galore_adamw8bit_per_layer       # Use GaLore AdamW optimizer with 8-bit precision per layer
