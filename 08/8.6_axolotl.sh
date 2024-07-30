#!/bin/bash

git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip3 install packaging
pip3 install accelerate transformers
pip3 install -e .[flash-attn, deepspeed]
pip3 install -U git+https://github.com/huggingface/peft.git

# download configuration
wget https://gist.githubusercontent.com/mlabonne/8055f6335e2b85f082c8c75561321a66/raw/93915a9563fcfff8df9a81fc0cdbf63894465922/EvolCodeLlama-7b.yaml

# launch finetune script
accelerate launch scripts/finetune.py EvolCodeLlama-7b.yaml

# The QLoRA adapter should already be uploaded to the Hugging Face Hub. However, you can also merge the base Code Llama model with this adapter and push the merged model.

# download script
wget https://gist.githubusercontent.com/mlabonne/a3542b0519708b8871d0703c938bba9f/raw/60abc5afc07f9d843bc23d56f4e0b7ab072c4a62/merge_peft.py

# merge model after fine-tuning
python merge_peft.py \
        --base_model=codellama/CodeLlama-7b-hf \
        --peft_model=./qlora-out \
        --hub_id=EvolCodeLlama-7b