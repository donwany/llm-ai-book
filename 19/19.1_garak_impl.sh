#!/bin/bash

# Setup Environment and Install Packages

# Install garak
python -m pip install -U garak

# Create and activate a new conda environment for garak
conda create --name garak "python>=3.10,<=3.12"
conda activate garak

# Clone the garak repository
gh repo clone leondz/garak
cd garak

# Install the required packages
python -m pip install -r requirements.txt

# Set Environment Variables and Run Probes
# Set the OpenAI API key
export OPENAI_API_KEY="sk-123XXXXXXXXXXXX"

# Run garak probes for OpenAI model
python3 -m garak --model_type openai --model_name gpt-3.5-turbo --probes encoding

# Run garak probes for Hugging Face model
python3 -m garak --model_type huggingface --model_name gpt2 --probes dan.Dan_11_0
