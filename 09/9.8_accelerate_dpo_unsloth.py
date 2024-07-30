import torch
from transformers import TrainingArguments
from trl import DPOTrainer
from unsloth import FastLanguageModel

# Configuration
max_seq_length = 2048  # Supports automatic RoPE Scaling, so choose any number.
train_dataset = "..."

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/zephyr-sft",
    max_seq_length=max_seq_length,
    dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit=True,  # Use 4bit quantization to reduce memory usage. Can be False.
    # token="hf_...", # use if using gated models like meta-llama/Llama-2-7b-hf
)

# Model patching and adding fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # Dropout = 0 is currently optimized
    bias="none",     # Bias = "none" is currently optimized
    use_gradient_checkpointing=True,
    random_state=3407,
)

# Set up training arguments
training_args = TrainingArguments(output_dir="./output")

# Initialize DPOTrainer
dpo_trainer = DPOTrainer(
    model,
    model_ref=None,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Train the model using DPO
dpo_trainer.train()
