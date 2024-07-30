#!/bin/bash

# Export your OpenAI API key
export OPENAI_API_KEY="your_api_key_here"

# Clone the AgentVerse repository with minimal depth
git clone https://github.com/OpenBMB/AgentVerse.git --depth 1

# Change directory to the cloned repository
cd AgentVerse

# Install the AgentVerse package in editable mode
pip install -e .

# Upgrade the agentverse package
pip install -U agentverse

# Run the simulation task via CLI
agentverse-simulation --task simulation/nlp_classroom_9players

# Run the simulation task via GUI
agentverse-simulation-gui --task simulation/nlp_classroom_9players

# Visit http://127.0.0.1:7860/ to view the classroom environment
