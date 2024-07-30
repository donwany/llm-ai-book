#!/bin/bash

conda create -n finetuning python=3.11 -y
conda activate finetuning
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=xxxxxxxxxx

pip install datasets transformers bitsandbytes sentencepiece accelerate loralib peft pillow torch torchvision hf_transfer