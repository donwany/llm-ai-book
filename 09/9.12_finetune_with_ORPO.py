# pip install -U transformers datasets accelerate peft trl bitsandbytes wandb
# Import necessary libraries
import gc
import os
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format

wb_token = "..."
# Log into Weights & Biases using the provided token
wandb.login(key=wb_token)
# Check the GPU compatibility for attention mechanism
if torch.cuda.get_device_capability()[0] >= 8:
    # pip install -qqq flash-attn attn_implementation="flash_attention_2"
    torch_dtype = torch.bfloat16  # Use bfloat16 if GPU is compatible
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16  # Otherwise, use float16
# Define the base model and the new model names
base_model = "meta-llama/Meta-Llama-3-8B"
new_model = "OrpoLlama-3-8B"
# Configuration for loading the model in 4-bit precision
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)
# Configuration for LoRA (Low-Rank Adaptation)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
# Load the tokenizer from the pre-trained model
tokenizer = AutoTokenizer.from_pretrained(base_model)
# Load the pre-trained model with the specified configuration
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)
# Set up chat format for the model and tokenizer
model, tokenizer = setup_chat_format(model, tokenizer)
# Prepare the model for k-bit training
model = prepare_model_for_kbit_training(model)
# Load and preprocess the dataset
dataset_name = "mlabonne/orpo-dpo-mix-40k"
dataset = load_dataset(dataset_name, split="all")
dataset = dataset.shuffle(seed=42).select(range(100))


# Function to format the chat template
def format_chat_template(row):
    row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
    row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
    return row


# Apply the formatting function to the dataset
dataset = dataset.map(
    format_chat_template,
    num_proc=os.cpu_count(),
)
# Split the dataset into training and testing sets
dataset = dataset.train_test_split(test_size=0.01)
# Configuration for ORPO training
orpo_args = ORPOConfig(
    learning_rate=8e-6,
    beta=0.1,
    lr_scheduler_type="linear",
    max_length=1024,
    max_prompt_length=512,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    report_to="wandb",
    output_dir="./outputs/",
)
# Create an ORPO trainer
trainer = ORPOTrainer(
    model=model,
    args=orpo_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    tokenizer=tokenizer,
)
# Train the model
trainer.train()
# Save the trained model
trainer.save_model(new_model)

# Clean up memory
del trainer, model
gc.collect()
torch.cuda.empty_cache()

# Reload the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Set up chat format again for the reloaded model and tokenizer
model, tokenizer = setup_chat_format(model, tokenizer)
# Merge the adapter weights with the base model
model = PeftModel.from_pretrained(model, new_model)
model = model.merge_and_unload()
# Push the final model and tokenizer to the Hugging Face hub
model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)