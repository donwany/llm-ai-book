#!/bin/bash

git clone https://github.com/dvlab-research/LongLoRA.git
cd LongLoRA
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# Fine-tuning
torchrun --nproc_per_node=8 fine-tune.py  \
            --model_name_or_path path_to/Llama-2-7b-hf \
            --bf16 True \
            --output_dir path_to_saving_checkpoints \
            --cache_dir path_to_cache \
            --model_max_length 8192 \
            --use_flash_attn True \
            --low_rank_training False \
            --num_train_epochs 1  \
            --per_device_train_batch_size 1  \
            --per_device_eval_batch_size 2  \
            --gradient_accumulation_steps 8  \
            --evaluation_strategy "no"  \
            --save_strategy "steps" \
            --save_steps 1000  \
            --save_total_limit 2 \
            --learning_rate 2e-5 \
            --weight_decay 0.0 \
            --warmup_steps 20  \
            --lr_scheduler_type "constant_with_warmup" \
            --logging_steps 1 \
            --deepspeed "ds_configs/stage2.json" \
            --tf32 True \
            --max_steps 1000

# supervised fine-tuning
torchrun --nproc_per_node=8 supervised-fine-tune.py  \
            --model_name_or_path path_to_Llama2_chat_models \
            --bf16 True \
            --output_dir path_to_saving_checkpoints \
            --model_max_length 16384 \
            --use_flash_attn True \
            --data_path LongAlpaca-16k-length.json \
            --low_rank_training True \
            --num_train_epochs 5  \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 2  \
            --gradient_accumulation_steps 8 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 98  \
            --save_total_limit 2 \
            --learning_rate 2e-5  \
            --weight_decay 0.0  \
            --warmup_steps 20 \
            --lr_scheduler_type "constant_with_warmup" \
            --logging_steps 1 \
            --deepspeed "ds_configs/stage2.json" \
            --tf32 True

# merge lora weights
python3 merge_lora_weights_and_save_hf_model.py \
            --base_model path_to/Llama-2-7b-hf \
            --peft_model path_to_saving_checkpoints \
            --context_size 8192 \
            --save_path path_to_saving_merged_model

# Inference
python3 inference.py  \
            --base_model /data/models/LongAlpaca-13B \
            --question "Why doesn't Professor Snape seem to like Harry?" \
            --context_size 32768 \
            --max_gen_len 512 \
            --flash_attn True \
            --material "materials/Harry Potter and the Philosophers Stone_section2.txt"