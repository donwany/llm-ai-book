#!/bin/bash

git clone https://github.com/THUDM/CogVLM.git
cd CogVLM/basic_demo
# CUDA >= 11.8
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# ------------------- Run CLI  -------------------
# CogAgent
python cli_demo_sat.py --from_pretrained cogagent-chat \
    --version chat \
    --bf16 \
    --stream_chat

python cli_demo_sat.py --from_pretrained cogagent-vqa --version chat_old --bf16  --stream_chat

# CogVLM
python cli_demo_sat.py --from_pretrained cogvlm-chat --version chat_old --bf16  --stream_chat

python cli_demo_sat.py \
    --from_pretrained cogvlm-grounding-generalist \
    --version base --bf16 --stream_chat