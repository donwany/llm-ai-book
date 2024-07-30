from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# Load the model
model_name_or_path = "meta-llama/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=True)
# Create LoRA config for the fine-tuning
peft_config = LoraConfig(
    nit_lora_weights="gaussian",
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=32,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['query_key_value']  # Optional: You can target specific layers using this
)
# Create a model ready for LoRA fine-tuning
model = get_peft_model(model, peft_config)
# Print trainable parameters
model.print_trainable_parameters()

# trainable params: 9,437,184
# || all params: 6,931,162,432
# || trainable%: 0.13615586263611604
