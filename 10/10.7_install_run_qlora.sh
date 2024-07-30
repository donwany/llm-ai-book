#!/bin/bash

git clone https://github.com/artidoro/qlora.git
pip install -U -r requirements.txt

cd qlora

python qlora.py --model_name_or_path <path_or_name>

# For models larger than 13B
python qlora.py --learning_rate 0.0001 --model_name_or_path <path_or_name>

# fine-tune using qlora
wget https://github.com/artidoro/qlora/blob/main/scripts/finetune.sh
# fine-tune using qlora with guanaco65B
wget https://github.com/artidoro/qlora/blob/main/scripts/finetune_guanaco_65b.sh
# generate inference
wget https://github.com/artidoro/qlora/blob/main/scripts/generate.sh