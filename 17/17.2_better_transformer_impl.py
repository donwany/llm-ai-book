# Import necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Define the quantization configuration for 4-bit precision
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Load the model in 4-bit precision
    bnb_4bit_compute_dtype=torch.float16  # Specify the computation data type
)

# Initialize the tokenizer with the pre-trained model
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

# Load the pre-trained model for causal language modeling with the quantization configuration
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", quantization_config=quantization_config)

# Enable the BetterTransformer module for optimized performance
model = model.to_bettertransformer()

# Define input text for generation
input_text = "Hello my dog is cute and"

# Tokenize the input text and move tensors to GPU if available
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# Generate output text based on the input
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    outputs = model.generate(**inputs)

# Decode the generated output and print it without special tokens
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
