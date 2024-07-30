#!/bin/bash

# Install litgpt and its dependencies
# Option 1: Install via pip
pip install 'litgpt[all]'

# Option 2: Install from source
git clone https://github.com/Lightning-AI/litgpt
cd litgpt
pip install -e '.[all]'

# 1) Download a pretrained model
litgpt download --repo_id mistralai/Mistral-7B-Instruct-v0.2

# 2) Chat with the downloaded model
litgpt chat \
  --checkpoint_dir checkpoints/mistralai/Mistral-7B-Instruct-v0.2
# >> Prompt: What do Llamas eat?

# Finetune an LLM
# 1) Download a pretrained model
litgpt download --repo_id microsoft/phi-2

# 2) Finetune the model with LoRA
litgpt finetune lora \
  --checkpoint_dir checkpoints/microsoft/phi-2 \
  --data Alpaca2k \
  --out_dir out/phi-2-lora

# 3) Chat with the fine-tuned model
litgpt chat \
  --checkpoint_dir out/phi-2-lora/final

# Train an LLM from scratch on your own data via pretraining
# 1) Download a pretrained model
litgpt download --repo_id microsoft/phi-2

# 2) Pretrain the model
litgpt pretrain \
  --initial_checkpoint_dir checkpoints/microsoft/phi-2 \
  --data Alpaca2k \
  --out_dir out/custom-phi-2

# Evaluate the model
litgpt evaluate \
  --checkpoint_dir checkpoints/microsoft/phi-2 \
  --batch_size 16 \
  --tasks "hellaswag,gsm8k,truthfulqa_mc2,mmlu,winogrande,arc_challenge"

# Generate text using the model
litgpt generate base \
  --prompt "Hello, my name is" \
  --checkpoint_dir checkpoints/stabilityai/stablelm-base-alpha-3b

# Quantize the model
pip install bitsandbytes

litgpt generate base \
  --quantize bnb.nf4 \
  --checkpoint_dir checkpoints/tiiuae/falcon-7b \
  --precision bf16-true \
  --max_new_tokens 256

# Example output
# Time for inference 1: 6.80 sec total, 37.62 tokens/sec
# Memory used: 5.72 GB
