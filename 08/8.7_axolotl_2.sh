#!/bin/bash

# preprocess datasets - optional but recommended
CUDA_VISIBLE_DEVICES="0" python \
        -m axolotl.cli.preprocess examples/openllama-3b/lora.yml

# finetune lora
accelerate launch -m axolotl.cli.train examples/openllama-3b/lora.yml

# inference
accelerate launch \
        -m axolotl.cli.inference examples/openllama-3b/lora.yml
        --lora_model_dir="./lora-out"

# gradio
accelerate launch \
        -m axolotl.cli.inference examples/openllama-3b/lora.yml
        --lora_model_dir="./lora-out" \
        --gradio

# remote yaml files - the yaml config can be hosted on a public URL
# Note: the yaml config must directly link to the **raw** yaml

accelerate launch \
        -m axolotl.cli.train \
        https://raw.githubusercontent.com/OpenAccess-AI-Collective/axolotl/main/examples/openllama-3b/lora.yml

# Pretrained LoRA
python -m axolotl.cli.inference examples/your_config.yml \
        --lora_model_dir="./lora-output-dir"

# Full weights finetune:
python -m axolotl.cli.inference examples/your_config.yml \
        --base_model="./completed-model"

# Full weights finetune w/ a prompt from a text file:
cat /tmp/prompt.txt | python -m $ axolotl.cli.inference examples/your_config.yml \
      --base_model="./completed-model" \
      --prompter=None \
      --load_in_8bit=True

# Using gradio hosting
python -m axolotl.cli.inference examples/your_config.yml --gradio

# Merge LoRA to base
python3 -m axolotl.cli.merge_lora your_config.yml \
        --lora_model_dir="./completed-model"

# Using Docker
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
# sudo usermod -aG docker $USER # Add current user to docker
newgrp docker
docker run --gpus '"all"' --rm -it winglian/axolotl:main-latest
accelerate launch -m axolotl.cli.train examples/openllama-3b/qlora.yml

# Upload model to huggingface: install huggingace cli
huggingface-cli upload <USERNAME>/MY-MODELNAME qlora-out