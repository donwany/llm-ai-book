#!/bin/bash

# --------- Installation and Setup ---------
# Install the 'trl' library for reinforcement learning with transformers
pip install trl

# Install the latest version of 'trl' from GitHub repository
pip install git+https://github.com/huggingface/trl.git

# Clone the 'trl' GitHub repository for local access
git clone https://github.com/huggingface/trl.git

# Download the example script for training LLaVA models with VSFT (Vanilla Supervised Fine-Tuning)
wget https://github.com/huggingface/trl/blob/main/examples/scripts/vsft_llava.py

# --------- Regular Training ---------
# Run the training script for the LLaVA model without PEFT (Parameter-Efficient Fine-Tuning)
python examples/scripts/vsft_llava.py \
    --dataset_name="HuggingFaceH4/llava-instruct-mix-vsft" \  # Dataset for training
    --model_name_or_path="llava-hf/llava-1.5-7b-hf" \  # Pre-trained model to fine-tune
    --report_to="wandb" \  # Reporting to Weights & Biases for tracking
    --learning_rate=1.4e-5 \  # Learning rate for training
    --per_device_train_batch_size=8 \  # Batch size per device (GPU)
    --gradient_accumulation_steps=1 \  # Number of gradient accumulation steps
    --output_dir="data/vsft-llava-1.5-7b-hf" \  # Directory to save model checkpoints
    --logging_steps=5 \  # Frequency of logging
    --num_train_epochs=1 \  # Number of training epochs
    --push_to_hub \  # Push the trained model to Hugging Face Hub
    --gradient_checkpointing \  # Enable gradient checkpointing to save memory
    --remove_unused_columns=False \  # Do not remove unused columns in the dataset
    --torch_dtype=float16 \  # Use float16 precision for training
    --fp16=True  # Enable mixed precision training

# --------- Training with PEFT (Parameter-Efficient Fine-Tuning) ---------
# Run the training script for the LLaVA model with PEFT
python examples/scripts/vsft_llava.py \
    --dataset_name="HuggingFaceH4/llava-instruct-mix-vsft" \  # Dataset for training
    --model_name_or_path="llava-hf/llava-1.5-7b-hf" \  # Pre-trained model to fine-tune
    --report_to="wandb" \  # Reporting to Weights & Biases for tracking
    --learning_rate=1.4e-5 \  # Learning rate for training
    --per_device_train_batch_size=8 \  # Batch size per device (GPU)
    --gradient_accumulation_steps=1 \  # Number of gradient accumulation steps
    --output_dir="data/vsft-llava-1.5-7b-hf" \  # Directory to save model checkpoints
    --logging_steps=5 \  # Frequency of logging
    --num_train_epochs=1 \  # Number of training epochs
    --push_to_hub \  # Push the trained model to Hugging Face Hub
    --gradient_checkpointing \  # Enable gradient checkpointing to save memory
    --remove_unused_columns=False \  # Do not remove unused columns in the dataset
    --torch_dtype=float16 \  # Use float16 precision for training
    --fp16=True \  # Enable mixed precision training
    --use_peft=True \  # Enable Parameter-Efficient Fine-Tuning
    --lora_r=64 \  # Rank of LoRA (Low-Rank Adaptation) matrices
    --lora_alpha=16 \  # Scaling factor for LoRA
    --lora_target_modules="all-linear"  # Target modules for LoRA application

