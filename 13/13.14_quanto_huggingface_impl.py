# Import necessary modules from the transformers library
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig

# Define the model ID for the pre-trained model
model_id = "facebook/opt-125m"

# Load the tokenizer associated with the specified model
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Configure quantization settings to use int8 for model weights
quantization_config = QuantoConfig(weights="int8")

# Load the model with the quantization configuration
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config
)

# Notebook Example:
# https://colab.research.google.com/drive/16CXfVmtdQvciSh9BopZUDYcmXCDpvgrT?usp=sharing
