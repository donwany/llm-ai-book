#!/bin/bash

# Clone the agents repository
git clone https://github.com/aiwaves-cn/agents.git

# Change directory to the cloned repository
cd agents

# Install the agents package in editable mode
pip install -e .

# Install the ai-agents package
pip install ai-agents

# Change directory to the examples folder
cd examples

# Run a single agent using the specified target agent configuration
python run.py --agent Single_Agent/{target_agent}/config.json

# Run multiple agents using the specified target agent configuration
python run.py --agent Multi_Agent/{target_agent}/config.json

# Run the Gradio interface for the single agent
python Single_Agent/run_gradio.py --agent Single_Agent/{target_agent}/config.json
