#!/bin/bash

# --------- Installation and Setup ---------
# Install the `lmms-eval` package
pip install lmms-eval

# Clone the `lmms-eval` repository from GitHub
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval

# Navigate into the `lmms-eval` directory
cd lmms-eval

# Install the `lmms-eval` package in editable mode
pip install -e .

# Alternatively, install `lmms-eval` directly from the GitHub repository
pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

# --------- Evaluating LLaVA on MME ---------
# Launch evaluation using `accelerate` with 8 processes
# Evaluate LLaVA model on the MME (MultiModal Evaluation) task
accelerate launch --num_processes=8 -m lmms_eval \
    --model llava \  # Model type to use
    --model_args pretrained="liuhaotian/llava-v1.5-7b" \  # Pre-trained model checkpoint
    --tasks mme \  # Task to evaluate (MME)
    --batch_size 1 \  # Batch size for evaluation
    --log_samples \  # Enable logging of samples
    --log_samples_suffix llava_v1.5_mme \  # Suffix for log files
    --output_path ./logs/  # Directory to save output logs

# --------- Evaluating LLaVA on Multiple Datasets ---------
# Launch evaluation using `accelerate` with 8 processes
# Evaluate LLaVA model on multiple tasks (MME and MMBench English)
accelerate launch --num_processes=8 -m lmms_eval \
    --model llava \  # Model type to use
    --model_args pretrained="liuhaotian/llava-v1.5-7b" \  # Pre-trained model checkpoint
    --tasks mme,mmbench_en \  # Tasks to evaluate (MME and MMBench English)
    --batch_size 1 \  # Batch size for evaluation
    --log_samples \  # Enable logging of samples
    --log_samples_suffix llava_v1.5_mme_mmbenchen \  # Suffix for log files
    --output_path ./logs/  # Directory to save output logs

# --------- Evaluating Other LLaVA Variants ---------
# Launch evaluation with a different model variant using `accelerate` with 8 processes
# Evaluate LLaVA model with a Mistral instruction template
accelerate launch --num_processes=8 -m lmms_eval \
    --model llava \  # Model type to use
    --model_args pretrained="liuhaotian/llava-v1.6-mistral-7b,conv_template=mistral_instruct" \  # Pre-trained model and conv_template
    --tasks mme,mmbench_en \  # Tasks to evaluate (MME and MMBench English)
    --batch_size 1 \  # Batch size for evaluation
    --log_samples \  # Enable logging of samples
    --log_samples_suffix llava_v1.5_mme_mmbenchen \  # Suffix for log files
    --output_path ./logs/  # Directory to save output logs

# Launch evaluation with another variant using `accelerate` with 8 processes
# Evaluate LLaVA model with a Mistral direct template
accelerate launch --num_processes=8 -m lmms_eval \
    --model llava \  # Model type to use
    --model_args pretrained="liuhaotian/llava-v1.6-34b,conv_template=mistral_direct" \  # Pre-trained model and conv_template
    --tasks mme,mmbench_en \  # Tasks to evaluate (MME and MMBench English)
    --batch_size 1 \  # Batch size for evaluation
    --log_samples \  # Enable logging of samples
    --log_samples_suffix llava_v1.5_mme_mmbenchen \  # Suffix for log files
    --output_path ./logs/  # Directory to save output logs

# --------- Evaluation from Predefined Configuration ---------
# Launch evaluation using a predefined configuration file with `accelerate`
# This allows for the evaluation of multiple models and datasets as specified in the configuration
accelerate launch --num_processes=8 -m lmms_eval \
    --config example_eval.yaml  # Configuration file specifying evaluation parameters
