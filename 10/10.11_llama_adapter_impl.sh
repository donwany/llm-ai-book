#!/bin/bash

conda create -n llama_adapter -y python=3.8
conda activate llama_adapter
# ------- install pytorch -------------
conda install pytorch cudatoolkit -c pytorch -y
# ------ install dependency and llama-adapter --------
pip install -r requirements.txt
pip install -e .

# fine-tune scripts
wget https://github.com/OpenGVLab/LLaMA-Adapter/blob/main/alpaca_finetuning_v1/finetuning.py
wget https://github.com/OpenGVLab/LLaMA-Adapter/blob/main/alpaca_finetuning_v1/finetuning.sh

# Training
cd alpaca_finetuning_v1

torchrun --nproc_per_node 8 finetuning.py \
             --model Llama7B_adapter \
             --llama_model_path $TARGET_FOLDER/ \
             --data_path $DATA_PATH/alpaca_data.json \
             --adapter_layer 30 \
             --adapter_len 10 \
             --max_seq_len 512 \
             --batch_size 4 \
             --epochs 5 \
             --warmup_epochs 2 \
             --blr 9e-3 \
             --weight_decay 0.02 \
             --output_dir ./checkpoint/

# Inference
torchrun --nproc_per_node 1 example.py \
        --ckpt_dir $TARGET_FOLDER/model_size\
        --tokenizer_path $TARGET_FOLDER/tokenizer.model \
        --adapter_path $ADAPTER_PATH