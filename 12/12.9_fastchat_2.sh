#!/bin/bash

# --------- Setup and Data Preparation ---------
# Clone the FastChat repository from GitHub
git clone https://github.com/lm-sys/FastChat.git

# Navigate into the FastChat directory
cd FastChat

# Install dependencies including model workers and LLM judges
pip install -e ".[model_worker,llm_judge]"

# Download pre-generated data for MT-bench
python3 download_mt_bench_pregenerated.py

# After downloading, view the data locally
python3 qa_browser.py --share

# --------- Model Evaluation on MT-bench ---------
# Step 1: Generate model answers to MT-bench questions
# Replace [MODEL-PATH] with the path to your model and [MODEL-ID] with the model identifier
python gen_model_answer.py --model-path [MODEL-PATH] --model-id [MODEL-ID]
# Example command:
python gen_model_answer.py --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5
# The answers will be saved to `data/mt_bench/model_answer/[MODEL-ID].jsonl`
# You can also specify `--num-gpus-per-model`, `--num-gpus-total` for GPU settings

# Step 2: Generate GPT-4 judgments
# Set your OpenAI API key
export OPENAI_API_KEY=XXXXXX
# Replace [LIST-OF-MODEL-ID] with a list of model identifiers and specify the number of concurrent API calls
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call]
# Example command:
python gen_judgment.py --model-list vicuna-13b-v1.3 alpaca-13b llama-13b claude-v1 gpt-3.5-turbo gpt-4 --parallel 2
# The judgments will be saved to `data/mt_bench/model_judgment/gpt-4_single.jsonl`

# Step 3: Show MT-bench scores
# Replace the list with your model identifiers
python show_result.py --model-list vicuna-13b-v1.3 alpaca-13b llama-13b claude-v1 gpt-3.5-turbo gpt-4
# To show all scores (if no specific model list is provided)
python show_result.py
