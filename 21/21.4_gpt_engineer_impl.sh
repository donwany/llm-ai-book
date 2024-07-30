#!/bin/bash

# Install necessary package
python -m pip install gpt-engineer

# Export OpenAI API Key
export OPENAI_API_KEY=[your api key]

# Clone the GPT Engineer repository
git clone https://github.com/gpt-engineer-org/gpt-engineer.git

# Change directory to the cloned repository
cd gpt-engineer

# Run the GPT Engineer tool with specified parameters
gpte projects/example-vision gpt-4-vision-preview \
--prompt_file prompt/text --image_directory prompt/images -i
