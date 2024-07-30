#!/bin/bash

# create: Config.yaml file
models:
  - model: mistralai/Mistral-7B-v0.1
    # no parameters necessary for base model
  - model: OpenPipe/mistral-ft-optimized-1218
    parameters:
      density: 0.5
      weight: 0.5
  - model: mlabonne/NeuralHermes-2.5-Mistral-7B
    parameters:
      density: 0.5
      weight: 0.3
merge_method: ties
base_model: mistralai/Mistral-7B-v0.1
parameters:
  normalize: true
dtype: float16

# --------------------------------------------------

# Clone the repository
git clone https://github.com/cg123/mergekit.git
cd mergekit
pip install -e .

# Usage
mergekit-yaml config.yaml ./output_folder \
    --allow-crimes \  # Allow mixing architectures
    --copy-tokenizer \  # Copy a tokenizer to the output
    --out-shard-size 1B \  # Number of parameters per output shard
    --low-cpu-memory \  # Store results and intermediate values on GPU. Useful if VRAM > RAM
    --write-model-card \  # Output README.md containing details of the merge
    --lazy-unpickle  # Experimental lazy unpickler for lower memory usage

# login to huggingface
huggingface-cli login --token=<hf...>

# upload your model
huggingface-cli upload <hf_username>/my-cool-model ./output-model
