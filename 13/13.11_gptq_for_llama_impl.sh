#!/bin/bash

# Create and activate a conda environment for GPTQ
# conda create --name gptq python=3.9 -y
# conda activate gptq

# Install PyTorch and related packages with CUDA support
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Alternatively, if conda installation has issues, use pip with python3.9:
# pip3 install torch torchvision torchaudio

# Clone the GPTQ-for-LLaMa repository and navigate into the directory
# git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa
# cd GPTQ-for-LLaMa

# Install required dependencies
# pip install -r requirements.txt

# Convert LLaMA weights to Hugging Face format
python convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights \
    --model_size 7B \
    --output_dir ./llama-hf

# -------- Benchmark language generation with 4-bit LLaMA-7B:--------

# Save compressed model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
python llama.py ${MODEL_DIR} c4 \
--wbits 4 \
--true-sequential \
--act-order \
--groupsize 128 \
--save llama7b-4bit-128g.pt

# Or save compressed `.safetensors` model
python llama.py ${MODEL_DIR} c4 \
--wbits 4 \
--true-sequential \
--act-order \
--groupsize 128 \
--save_safetensors llama7b-4bit-128g.safetensors

# Benchmark generating a 2048 token sequence with the saved model
python llama.py ${MODEL_DIR} c4 \
  --wbits 4 \
  --groupsize 128 \
  --load llama7b-4bit-128g.pt \
  --benchmark 2048 \
  --check

# Benchmark FP16 baseline, note that the model will be split across all listed GPUs
python llama.py ${MODEL_DIR} c4 --benchmark 2048 --check

# Model inference with the saved model
python llama_inference.py ${MODEL_DIR} \
  --wbits 4 \
  --groupsize 128 \
  --load llama7b-4bit-128g.pt \
  --text "this is llama"

# Model inference with the saved model using safetensors loaded directly to GPU
python llama_inference.py ${MODEL_DIR} \
  --wbits 4 \
  --groupsize 128 \
  --load llama7b-4bit-128g.safetensors \
  --text "this is llama" --device=0

# Model inference with the saved model with offload (note: this is very slow)
python llama_inference_offload.py ${MODEL_DIR} \
  --wbits 4 \
  --groupsize 128 \
  --load llama7b-4bit-128g.pt \
  --text "this is llama" \
  --pre_layer 16

# Note: Generating 45 tokens on a single RTX3090 with LLaMa-65B (pre_layer set to 50) takes about 180 seconds.
