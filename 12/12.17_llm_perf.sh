#!/bin/bash

# --------- Setup Environment ---------
# Clone the LLMPerf repository from GitHub
git clone https://github.com/ray-project/llmperf.git
# Navigate to the LLMPerf directory
cd llmperf
# Install the LLMPerf package in editable mode
pip install -e .

# --------- OPENAI API Benchmark ---------
# Set OpenAI API key and base URL
export OPENAI_API_KEY=secret_abcdefg
export OPENAI_API_BASE="https://api.endpoints.anyscale.com/v1"
# Run benchmark with OpenAI API
python token_benchmark_ray.py \
    --model "meta-llama/Llama-2-7b-chat-hf" \
    --mean-input-tokens 550 \
    --stddev-input-tokens 150 \
    --mean-output-tokens 150 \
    --stddev-output-tokens 10 \
    --max-num-completed-requests 2 \
    --timeout 600 \
    --num-concurrent-requests 1 \
    --results-dir "result_outputs" \
    --llm-api openai \
    --additional-sampling-params '{}'

# --------- HUGGINGFACE API Benchmark ---------
# Set HuggingFace API key and base URL
export HUGGINGFACE_API_KEY="YOUR_HUGGINGFACE_API_KEY"
export HUGGINGFACE_API_BASE="YOUR_HUGGINGFACE_API_ENDPOINT"
# Run benchmark with HuggingFace API
python token_benchmark_ray.py \
    --model "huggingface/meta-llama/Llama-2-7b-chat-hf" \
    --mean-input-tokens 550 \
    --stddev-input-tokens 150 \
    --mean-output-tokens 150 \
    --stddev-output-tokens 10 \
    --max-num-completed-requests 2 \
    --timeout 600 \
    --num-concurrent-requests 1 \
    --results-dir "result_outputs" \
    --llm-api "litellm" \
    --additional-sampling-params '{}'

# --------- ANTHROPIC API Benchmark ---------
# Set Anthropic API key
export ANTHROPIC_API_KEY=secret_abcdefg
# Run benchmark with Anthropic API
python token_benchmark_ray.py \
    --model "claude-2" \
    --mean-input-tokens 550 \
    --stddev-input-tokens 150 \
    --mean-output-tokens 150 \
    --stddev-output-tokens 10 \
    --max-num-completed-requests 2 \
    --timeout 600 \
    --num-concurrent-requests 1 \
    --results-dir "result_outputs" \
    --llm-api anthropic \
    --additional-sampling-params '{}'

# --------- TOGETHERAI API Benchmark ---------
# Set TogetherAI API key
export TOGETHERAI_API_KEY="YOUR_TOGETHER_KEY"
# Run benchmark with TogetherAI API
python token_benchmark_ray.py \
    --model "together_ai/togethercomputer/CodeLlama-7b-Instruct" \
    --mean-input-tokens 550 \
    --stddev-input-tokens 150 \
    --mean-output-tokens 150 \
    --stddev-output-tokens 10 \
    --max-num-completed-requests 2 \
    --timeout 600 \
    --num-concurrent-requests 1 \
    --results-dir "result_outputs" \
    --llm-api "litellm" \
    --additional-sampling-params '{}'

# --------- SAGEMAKER API Benchmark ---------
# Set AWS credentials and region
export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"
export AWS_SESSION_TOKEN="YOUR_SESSION_TOKEN"
export AWS_REGION_NAME="YOUR_ENDPOINTS_REGION_NAME"
# Run benchmark with SageMaker API
python llm_correctness.py \
    --model "llama-2-7b" \
    --llm-api "sagemaker" \
    --max-num-completed-requests 2 \
    --timeout 600 \
    --num-concurrent-requests 1 \
    --results-dir "result_outputs"
