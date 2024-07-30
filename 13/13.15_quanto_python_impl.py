# Install necessary packages
# pip install transformers==4.35.0 quanto==0.0.11 torch==2.1.1

def compute_model_sizes(model):
    """
    Compute the sizes of the model's parameters and the total size of the model.
    Args:
        model: The model whose parameters' sizes are to be computed.
    Returns:
        A tuple containing:
            - A dictionary with parameter names and their sizes in bytes.
            - The total size of the model in gigabytes.
    """
    module_sizes = {}
    total_size_bytes = 0
    for name, param in model.named_parameters():
        module_size_bytes = param.numel() * param.element_size()  # Size in bytes
        module_sizes[name] = module_size_bytes
        total_size_bytes += module_size_bytes
    total_size_gb = total_size_bytes * 1e-9  # Convert total size to gigabytes
    return module_sizes, total_size_gb


from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the pre-trained model and tokenizer
model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Generate text using the model
input_ids = tokenizer("Hello, what is your name?", return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Compute and print the model size before quantization
_, total_size_gb = compute_model_sizes(model)
print(f"Total size before quantization: {total_size_gb:.6f} GB")

# Inspect model weights before quantization (replace <LAYER> with the actual layer name)
# print(model.<LAYER>.layers[0].attention.dense.weight)

# Quantize the model weights using Quanto
from quanto import quantize, freeze
import torch

# Quantize model weights to 8 bits
quantize(model, weights=torch.int8, activations=None)
print("Model after quantization:")
print(model)

# Freeze the quantized model
freeze(model)

# Compute and print the model size after quantization
_, total_size_gb = compute_model_sizes(model)
print(f"Total size after quantization: {total_size_gb:.6f} GB")

# Inspect model weights after quantization (replace <LAYER> with the actual layer name)
# print(model.<LAYER>.layers[0].attention.dense.weight)

# Perform inference on the quantized model
input_ids = tokenizer("Hello, what is your name?", return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
