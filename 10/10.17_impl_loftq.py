# Clone the LoftQ repository and install the required dependencies
# git clone https://github.com/yxli2123/LoftQ.git
# cd LoftQ
# pip install -r requirements.txt

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Define the model identifier (MODEL_ID) from Hugging Face
MODEL_ID = "LoftQ/Mistral-7B-v0.1-4bit-64rank"

# Load the base model with specified quantization configuration
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,  # Set the data type for computations
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,  # Use 4-bit quantization to reduce memory usage
        bnb_4bit_compute_dtype=torch.bfloat16,  # Recommended data type for 4-bit quantization
        bnb_4bit_use_double_quant=False,  # Disable double quantization
        bnb_4bit_quant_type='nf4',  # Quantization type
    )
)

# Load the PEFT (Parameter-Efficient Fine-Tuning) model from the specified subfolder
peft_model = PeftModel.from_pretrained(
    base_model,
    MODEL_ID,
    subfolder="loftq_init",  # Subfolder containing initial LoftQ configurations
    is_trainable=True  # Set the model as trainable
)

# Fine-Tuning with LoftQ (Run this in your terminal or script)
# Fine-tuning command for training the model
# python train_gsm8k.py \
#     --model_name_or_path LoftQ/Llama-2-7b-hf-4bit-64rank \
#     --learning_rate 3e-4 --seed 11 \
#     --expt_name gsm8k_llama2_7b_4bit_64rank_loftq \
#     --output_dir exp_results/ \
#     --num_train_epochs 6 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" --weight_decay 0.1 \
#     --warmup_ratio 0.03 --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --do_train \
#     --report_to tensorboard

# Evaluation (Run this in your terminal or script)
# Evaluation command for testing the model
# python test_gsm8k.py \
#     --model_name_or_path LoftQ/Llama-2-7b-hf-4bit-64rank \
#     --batch_size 16
