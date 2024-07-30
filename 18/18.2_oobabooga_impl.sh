#!/bin/bash

# Run the start_linux.sh, start_windows.bat, start_macos.sh,
# or start_wsl.bat script depending on your OS.
# Select your GPU vendor when asked.
# Once the installation ends, browse to http://localhost:7860/?__theme=dark.
# Have fun!
# ----------------------------------------------------------------
conda install -y -c "nvidia/label/cuda-12.1.1" cuda-runtime
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
pip install -r requirements.txt
# START WEB UI
conda activate textgen
cd text-generation-webui

python server.py
# Then browse to:
# http://localhost:7860/?__theme=dark

# Download model here
python download-model.py organization/model


# Colab Notebook here:
# https://colab.research.google.com/github/oobabooga/text-generation-webui/blob/main/Colab-TextGen-GPU.ipynb