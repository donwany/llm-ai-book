#!/bin/bash

wget https://github.com/01-ai/Yi/blob/main/quantization/gptq/quant_autogptq.py

python quant_autogptq.py \
      --model /base_model \
      --output_dir /quantized_model \
      --trust_remote_code

# GPTQ Models
# https://huggingface.co/TheBloke/Yi-34B-GPTQ
# https://huggingface.co/TheBloke/CodeLlama-70B-Python-GPTQ
# https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ
# https://huggingface.co/TheBloke/llava-v1.5-13B-GPTQ