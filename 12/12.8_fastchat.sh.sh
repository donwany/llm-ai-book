#!/bin/bash

# --------- Installation ---------
# Clone the FastChat repository from GitHub
git clone https://github.com/lm-sys/FastChat.git

# Navigate into the FastChat directory
cd FastChat

# Install dependencies including model workers and LLM judges
pip install -e ".[model_worker,llm_judge]"

# Additional installations for matplotlib and tabulate
pip install matplotlib tabulate

# --------- Running FastChat ---------
# Run on a single GPU
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5

# Run on multiple GPUs (specify the number of GPUs)
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --num-gpus 2

# Run with CLI options for a specific device (e.g., Apple MPS) and load 8-bit weights
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --device mps --load-8bit

# Launch Gradio web server to interact with the model through a web interface
python3 -m fastchat.serve.gradio_web_server
