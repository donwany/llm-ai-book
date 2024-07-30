from hqq.core.quantize import *
from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
import torch

# Define quantization settings
quant_config = BaseQuantizeConfig(nbits=4, group_size=64)

# Replace a linear layer with a quantized HQQLinear layer
hqq_layer = HQQLinear(
    your_linear_layer,  # Replace with torch.nn.Linear layer or None
    quant_config=quant_config,  # Quantization configuration
    compute_dtype=torch.float16,  # Data type for computation
    device='cuda',  # Device to use
    initialize=True,  # Initialize quantization immediately (set to False to quantize later)
    del_orig=True  # If True, delete the original layer
)

# Model and settings
model_id = 'meta-llama/Llama-2-7b-chat-hf'
compute_dtype = torch.float16
device = 'cuda:0'

# Load the model on the CPU
model = HQQModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Quantize the model
model.quantize_model(
    quant_config=quant_config,
    compute_dtype=compute_dtype,
    device=device
)

# Save and load a quantized model
save_dir = 'path_to_save_quantized_model'  # Replace with your desired save directory

# Save the quantized model
model.save_quantized(save_dir=save_dir)

# Load the quantized model from local directory or Hugging Face Hub on a specific device
save_dir_or_hfhub = 'path_to_quantized_model'  # Replace with your model's path or Hugging Face Hub identifier
model = HQQModelForCausalLM.from_quantized(
    save_dir_or_hfhub,
    device='cuda'
)
