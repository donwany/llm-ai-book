#!/bin/bash

# Clone the Macaw-LLM repository from GitHub
git clone https://github.com/lyuchenyang/Macaw-LLM.git

# Change the current directory to the Macaw-LLM repository
cd Macaw-LLM

# Install required Python packages specified in the requirements file
pip install -r requirements.txt

# Install ffmpeg for handling audio and video processing
yum install ffmpeg -y

# Install NVIDIA's apex library for mixed-precision training
# Clone the apex repository from GitHub
git clone https://github.com/NVIDIA/apex.git

# Change directory to the cloned apex repository
cd apex

# Install apex using its setup script
python setup.py install

# Return to the previous directory
cd ..

# Train the model using the training script provided in the Macaw-LLM repository
./train.sh

# Perform inference using the inference script provided in the Macaw-LLM repository
./inference.sh
