from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Define the model ID and specify the device for computation
model_id = "meta-llama/LlamaGuard-7b"  # ID of the pre-trained model
device = "cuda"  # Use GPU (CUDA) if available

# Initialize the tokenizer using the specified pre-trained model
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the pre-trained causal language model with specific data type and device mapping
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)


def moderate(chat):
    """
    Generate a response based on the chat history.
    Tokenizes the chat input, generates a response using the model, and decodes the output.
    """
    # Apply the chat template to the input chat and convert to tensor format
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)

    # Generate a response from the model
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)

    # Calculate the length of the prompt to slice the generated output accordingly
    prompt_len = input_ids.shape[-1]

    # Decode the generated output to a human-readable string, skipping special tokens
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


# Example usage of the moderate function with a chat input
chat_history = [
    {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"},  # User query
    {"role": "assistant",
     "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate."}
    # Assistant response
]

# Generate a response based on the chat history
response = moderate(chat_history)
print(response)
