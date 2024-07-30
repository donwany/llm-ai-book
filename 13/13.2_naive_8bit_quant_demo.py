# --------- Install Dependencies ---------
# Install the bitsandbytes package for quantization
# pip install -q bitsandbytes>=0.39.0
# Install the accelerate library from Hugging Face
# pip install -q git+https://github.com/huggingface/accelerate.git
# Install the transformers library from Hugging Face
# pip install -q git+https://github.com/huggingface/transformers.git

# --------- Import Libraries ---------
import torch
from transformers import GPT2Model, GPT2Tokenizer

# --------- Device Setup ---------
# Set device to CPU for now
device = 'cpu'

# --------- Load Model and Tokenizer ---------
# Specify model ID
model_id = 'gpt2'
# Load pre-trained GPT-2 model and move to specified device
model = GPT2Model.from_pretrained(model_id).to(device)
# Load pre-trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_id)

# --------- Model Size ---------
# Calculate model size in bytes
model_size_bytes = model.config.total_params * 4  # Assuming 4 bytes per parameter
print(f"Model size: {model_size_bytes:,} bytes")

# --------- Extract and Quantize Weights ---------
# Extract weights of the first layer of the model
weights = model.transformer.h[0].attn.c_attn.weight.data
print(f"Original weights: {weights}")


# Define the absmax quantization function
def absmax_quantize(weights):
    absmax = weights.abs().max()
    scale = absmax / 127.0  # 127 for 8-bit quantization
    quantized_weights = (weights / scale).round().clamp(-128, 127)
    return quantized_weights, scale


# Define the zero-point quantization function
def zeropoint_quantize(weights):
    min_weight, max_weight = weights.min(), weights.max()
    scale = (max_weight - min_weight) / 255.0  # 255 for 8-bit quantization
    zero_point = -min_weight / scale
    quantized_weights = (weights / scale + zero_point).round().clamp(0, 255)
    return quantized_weights, scale


# Quantize weights using absmax quantization
weights_abs_quant, _ = absmax_quantize(weights)
print("\nAbsmax quantized weights:")
print(weights_abs_quant)

# Quantize weights using zero-point quantization
weights_zp_quant, _ = zeropoint_quantize(weights)
print(f"\nZero-point quantized weights: {weights_zp_quant}")
