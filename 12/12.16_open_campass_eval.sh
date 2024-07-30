#!/bin/bash

# --------- Setup Environment ---------
# Create a new Conda environment with Python 3.10 and necessary packages
# Note: Adjust package versions and channels as needed.
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
# Activate the newly created environment
conda activate opencompass

# --------- Clone and Install OpenCompass ---------
# Clone the OpenCompass repository from GitHub
git clone https://github.com/open-compass/opencompass opencompass
# Navigate to the OpenCompass directory
cd opencompass
# Install the OpenCompass package in editable mode
pip install -e .

# --------- Data Preparation ---------
# Download the dataset and unzip it into the data/ folder
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip

# --------- Evaluation ---------
# Run evaluation with default models and datasets
python run.py --models hf_llama_7b --datasets mmlu_ppl ceval_ppl

# Evaluate using a specific HuggingFace model
python run.py --datasets ceval_ppl mmlu_ppl \
    --hf-path huggyllama/llama-7b \  # Path to the HuggingFace model
    --model-kwargs device_map='auto' \  # Arguments for model construction
    --tokenizer-kwargs padding_side='left' truncation='left' use_fast=False \  # Arguments for tokenizer construction
    --max-out-len 100 \  # Maximum number of tokens generated
    --max-seq-len 2048 \  # Maximum sequence length the model can accept
    --batch-size 8 \  # Batch size for evaluation
    --no-batch-padding \  # Disable batch padding to avoid performance loss
    --num-gpus 1  # Number of GPUs to use (minimum required)
