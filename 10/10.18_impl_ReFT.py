# Install the pyreft library from the GitHub repository
# pip install git+https://github.com/stanfordnlp/pyreft.git

import torch
import transformers
from pyreft import *  # Import all functions and classes from the pyreft module

# Define the model name or path and device
model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"  # Model identifier from Hugging Face
device = "0"  # Specify the device ID to use (e.g., "0" for GPU 0)

# Load the base model from Hugging Face with specified dtype and device mapping
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,  # Use bfloat16 precision for the model
    device_map=device  # Map the model to the specified device
)

# Configure the Rank-1 Constant Reft for the model
reft_config = pyreft.ReftConfig(
    representations={
        "layer": 15,  # Target layer in the model for Reft
        "component": "block_output",  # Component within the layer
        "low_rank_dimension": 4,  # Low-rank dimension for Reft
        "intervention": pyreft.LoreftIntervention(
            embed_dim=model.config.hidden_size,  # Size of the embedding dimension
            low_rank_dimension=4  # Dimension for the low-rank intervention
        )
    }
)

# Apply the Reft configuration to the model
reft_model = pyreft.get_reft_model(model, reft_config)

# Set the Reft model to use GPU
reft_model.set_device("cuda")

# Print the number of trainable parameters
reft_model.print_trainable_parameters()

"""
Output:
Trainable intervention parameters: 32,772
Trainable model parameters: 0
Total model parameters: 6,738,415,616
Trainable parameters percentage: 0.000486%
"""
