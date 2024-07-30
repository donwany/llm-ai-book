# Install necessary packages
# pip install -qU transformers accelerate

from transformers import AutoTokenizer
import transformers
import torch

# Specify the model ID for the pre-trained model to be used
model = "<HF_USERNAME>/<MERGED-MODEL-NAME>"  # Replace with actual model name

# Define the chat messages with roles and contents
messages = [{"role": "user", "content": "What is a llm?"}]

# Initialize the tokenizer with the specified pre-trained model
tokenizer = AutoTokenizer.from_pretrained(model)

# Apply a chat template to the messages without tokenizing and add a generation prompt
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Create a text generation pipeline with the specified model
# Use float16 for faster computation on supported devices and map the model to available devices automatically
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Generate text based on the prompt with specified parameters
# max_new_tokens=256: Generate up to 256 new tokens
# do_sample=True: Use sampling instead of greedy decoding
# temperature=0.7: Control randomness in sampling (lower is less random)
# top_k=50: Consider only the top 50 tokens for sampling
# top_p=0.95: Use nucleus sampling to consider tokens that make up 95% of the probability mass
outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

# Print the generated text from the first output
print(outputs[0]["generated_text"])
