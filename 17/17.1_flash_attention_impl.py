import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the model ID
model_id = "tiiuae/falcon-7b"

# Load the tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model with specific configurations
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # Set the data type to bfloat16 for optimized performance
    # Note: The argument `attn_implementation` is not a valid argument for `AutoModelForCausalLM`
    # If using `flash_attention_2`, it must be supported by the specific model and framework
)

# Optional: Print model and tokenizer info to verify
print(f"Loaded model: {model_id}")
print(f"Tokenizer: {tokenizer}")
