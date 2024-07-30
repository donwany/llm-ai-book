from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the base model from Hugging Face Hub
model = AutoModelForCausalLM.from_pretrained("qwen/Qwen-1_8B-Chat")

# Load the tokenizer associated with the base model
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen-1_8B-Chat")

# Load the PEFT (Parameter Efficient Fine-Tuning) model with specified configuration
model = PeftModel.from_pretrained(model=model,
    model_id="<HF_USERNAME>/MY_MODEL_NAME")  # Replace <HF_USERNAME>/MY_MODEL_NAME with your model ID

# Define the query that will be sent to the language model
query_to_llm = "It was originally priced at $500 but it's on sale for 50% off. Can you tell me how much it will cost after the discount?"

# Encode the query into token IDs using the tokenizer
inputs = tokenizer.encode(query_to_llm, return_tensors="pt")

# Generate a response from the model using the encoded inputs
outputs = model.generate(inputs)

# Decode the generated token IDs back into a human-readable string
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the response from the model
print(response)
