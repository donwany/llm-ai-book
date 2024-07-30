#!/bin/bash

# Clone the repository
git clone https://github.com/ShishirPatil/gorilla.git

# Change directory to 'raft'
cd gorilla/raft

# Install required packages
pip install -r requirements.txt

# Run RAFT with specified parameters
python3 raft.py \
    --datapath PATH_TO_DATA \
    --output OUTPUT_PATH \
    --distractors 3 \
    --doctype pdf \
    --chunk_size 512 \
    --questions 5 \
    --openai_key YOUR_OPENAI_KEY

# Example usage
python3 raft.py \
    --datapath sample_data/United_States_PDF.pdf \
    --output ./sample_ds4 \
    --distractors 4 \
    --doctype pdf \
    --chunk_size 512 \
    --questions 5 \
    --openai_key OPENAI_KEY


# Evaluate RAFT performance
python3 eval.py \
    --question-file YOUR_EVAL_FILE.jsonl \
    --answer-file YOUR_ANSWER_FILE
