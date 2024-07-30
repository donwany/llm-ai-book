#!/bin/bash

# --------------- Step 1: Installation -----------------
# conda create -n taskweaver python=3.10
# conda activate taskweaver
# clone the repository
# git clone https://github.com/microsoft/TaskWeaver.git
# cd TaskWeaver
# install the requirements
# pip install -r requirements.txt

# ---------------- Step 2: Configure the LLMs ----------
# Configure the taskweaver_config.json file
{
  "llm.api_type": "openai",
  "llm.api_base": "https://api.openai.com/v1",
  "llm.api_key": "sk-xxxxxxxxxxxxxxxxxxxx",
  "llm.model": "gpt-4",
  "llm.response_format": "json_object"
}

# ---------------- Step 3: Start TaskWeaver ------------
python -m taskweaver -p ./project/
