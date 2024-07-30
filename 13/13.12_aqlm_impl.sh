#!/bin/bash

# Install the required package
# pip install aqlm[gpu,cpu]

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

# Quantized Model
quantized_model = AutoModelForCausalLM.from_pretrained(
    "ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf",
    trust_remote_code=True, torch_dtype="auto"
).cuda()

# Download model
hf_hub_download(repo_id="Vahe1994/AQLM", filename="data/name.pth", repo_type="dataset")

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype="auto")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

# Clone the AQLM repository
# git clone https://github.com/Vahe1994/AQLM.git

# Set environment variables for quantization
export CUDA_VISIBLE_DEVICES=0   # or e.g. 0,1,2,3
export MODEL_PATH=<PATH_TO_MODEL_ON_HUB>
export DATASET_PATH=<INSERT_DATASET_NAME_OR_PATH_TO_CUSTOM_DATA>
export SAVE_PATH=/path/to/save/quantized/model/
export WANDB_PROJECT=MY_AQ_EXPS
export WANDB_NAME=COOL_EXP_NAME

# Run the quantization script
python main.py $MODEL_PATH $DATASET_PATH \
    --nsamples=1024 \
    --val_size=128 \
    --num_codebooks=1 \
    --nbits_per_codebook=16 \
    --in_group_size=8 \
    --relative_mse_tolerance=0.01 \
    --finetune_batch_size=32 \
    --finetune_max_epochs=10 \
    --finetune_early_stop=3 \
    --local_batch_size=1 \
    --offload_activations \
    --wandb \
    --save $SAVE_PATH

# Set environment variables for evaluation
export CUDA_VISIBLE_DEVICES=0,1,2,3  # optional: select GPUs
export QUANTZED_MODEL=<PATH_TO_SAVED_QUANTIZED_MODEL_FROM_MAIN.py>
export MODEL_PATH=<INSERT_PATH_TO_ORIGINAL_MODEL_ON_HUB>
export DATASET=<INSERT_DATASET_NAME_OR_PATH_TO_CUSTOM_DATA>
export WANDB_PROJECT=MY_AQ_LM_EVAL
export WANDB_NAME=COOL_EVAL_NAME

# Run the evaluation script
python lmeval.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_PATH,dtype=float16,use_accelerate=True \
    --load $QUANTZED_MODEL \
    --tasks winogrande,piqa,hellaswag,arc_easy,arc_challenge \
    --batch_size 1

# Example Notebook here:
# https://colab.research.google.com/drive/1-xZmBRXT5Fm3Ghn4Mwa2KRypORXb855X?usp=sharing
# https://colab.research.google.com/drive/12GTp1FCj5_0SnnNQH18h_2XFh9vS_guX?usp=sharing