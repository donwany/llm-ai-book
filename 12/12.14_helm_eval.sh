#!/bin/bash

# --------- Setup ---------
# Create a virtual environment named 'crfm-helm' with Python 3.8 and pip
# conda create -n crfm-helm python=3.8 pip
# Activate the created environment
# conda activate crfm-helm
# Install the crfm-helm package within the environment
# pip install crfm-helm

# --------- Run Evaluations ---------
# Run the boolq evaluation on the model 'stanford-crfm/BioMedLM' using the default 'main' revision
helm-run \
    --run-entries boolq:model=stanford-crfm/BioMedLM \
    --enable-huggingface-models stanford-crfm/BioMedLM \
    --suite v1 \
    --max-eval-instances 10

# Run the boolq evaluation on 'stanford-crfm/BioMedLM' at the 'main' revision explicitly
helm-run \
    --run-entries boolq:model=stanford-crfm/BioMedLM@main \
    --enable-huggingface-models stanford-crfm/BioMedLM@main \
    --suite v1 \
    --max-eval-instances 10

# --------- Run Benchmark ---------
# Run a benchmark using a configuration file and specific suite
helm-run --conf-paths run_entries.conf --suite v1 --max-eval-instances 10

# Summarize the benchmark results
helm-summarize --suite v1

# Start a web server to display the benchmark results
helm-server
# Open your browser and navigate to http://localhost:8000/ to view the results
