#!/bin/bash

wget https://github.com/01-ai/Yi/blob/main/quantization/awq/quant_autoawq.py

python quant_autoawq.py \
        --model /base_model \
        --output_dir /quantized_model \
        --bits 4 \
        --group_size 128 \
        --trust_remote_code

# -------------------- Install ---------------------------
# git clone https://github.com/mit-han-lab/llm-awq
# cd llm-awq
# conda create -n awq python=3.10 -y
# conda activate awq
# pip install --upgrade pip
# pip install -e .

# Usage:
# https://github.com/mit-han-lab/llm-awq/blob/main/examples/convert_to_hf.py
# https://github.com/mit-han-lab/llm-awq/blob/main/scripts/llama2_example.sh

# AWQ Models
# https://huggingface.co/TheBloke/Yi-34B-AWQ
# https://huggingface.co/TheBloke/CodeLlama-70B-Instruct-AWQ
# https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ
# https://huggingface.co/TheBloke/LlamaGuard-7B-AWQ

# Usage:
#from transformers import AutoModelForCausalLM, AutoTokenizer
#model_name_or_path = "TheBloke/Yi-34B-AWQ"
#
#tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#model = AutoModelForCausalLM.from_pretrained(
#        model_name_or_path,
#        low_cpu_mem_usage=True,
#        device_map="cuda:0"
#)