# Install Causal Conv1d
# pip install causal-conv1d>=1.2.0

# Install Mamba
# pip install mamba-ssm

# Build from Source (if necessary): If you prefer to build from the source repository
# pip install .

# Handling PyTorch Versions: If you encounter issues with PyTorch versions, you can try passing --no-build-isolation to pip
# pip install --no-build-isolation mamba-ssm

# Additional Requirements: Ensure you have the following prerequisites
# Linux, NVIDIA GPU, PyTorch 1.12 or higher, CUDA 11.6 or higher

from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-790m-hf")
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-790m-hf")
input_ids = tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"]

out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
