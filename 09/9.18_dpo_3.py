# Load the saved model from SFT step as base model
from peft import AutoPeftModelForCausalLM
from transformers import training_args
from trl import DPOTrainer
import torch
from trl.commands.scripts.ppo_multi_adapter import script_args

model = AutoPeftModelForCausalLM.from_pretrained(
        script_args.model_name_or_path, # Location of saved SFT model
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        is_trainable=True,
    )

# Use the same saved model as the reference model for DPO
model_ref = AutoPeftModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,  # Same model as the main one
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
)

train_dataset = "..."
eval_dataset = "..."
tokenizer = "..."
peft_config = "..."

# Initialize DPOTrainer
dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,          # HF Trainer arguments
        beta=script_args.beta,       # Beta hyperparameter for DPO loss
        train_dataset=train_dataset, # Training dataset
        eval_dataset=eval_dataset,   # Evaluation dataset
        tokenizer=tokenizer,         # Tokenizer
        peft_config=peft_config,     # LoRA configuration
        beta=0.1,                    # temperature hyperparameter of DPO
    )

# Train DPOTrainer
dpo_trainer.train()
# Save the trained DPO model
dpo_trainer.save_model()