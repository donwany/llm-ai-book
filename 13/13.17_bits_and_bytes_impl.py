# Install the bitsandbytes library
# pip install bitsandbytes

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
import torch

# Define the quantization configuration for 4-bit precision
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable loading model in 4-bit precision
    bnb_4bit_compute_dtype=torch.bfloat16  # Use bfloat16 for computation
)

# Load the pre-trained model with 4-bit precision
model_id = "mistralai/Mistral-7B-v0.1"
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config
)

# Prepare the model for k-bit training (4-bit in this case)
model_4bit = prepare_model_for_kbit_training(model_4bit)

# Print confirmation of successful setup
print("Model loaded and prepared for 4-bit training.")
