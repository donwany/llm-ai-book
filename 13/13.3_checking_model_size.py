# --------- Import Libraries ---------
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --------- Set Random Seed ---------
# Ensure reproducibility
torch.manual_seed(0)

# --------- Device Setup ---------
# Set device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------- Load Model and Tokenizer ---------
# Specify model ID
model_id = 'gpt2'
# Load pre-trained GPT-2 model and move to specified device
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
# Load pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# --------- Print Model Size ---------
# Print model size in bytes
print(f"Model size: {model.get_memory_footprint():,} bytes")
# Model size: 510,342,192 bytes

# --------- Load Model in 8-Bits ---------
# Load the same model but in 8-bit precision
model_int8 = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    load_in_8bit=True,
)

# --------- Print Quantized Model Size ---------
# Print model size in bytes after quantization
print(f"Model size: {model_int8.get_memory_footprint():,} bytes")
# Model size: 176,527,896 bytes
