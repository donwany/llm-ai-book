from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

model_name = "NousResearch/llama-2-7b-chat-hf"
new_model = "llama-2-7b-miniguanaco"
# Reload the base model in FP16 precision
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,            # The name of the pre-trained model to load
    low_cpu_mem_usage=True, # Optimize memory usage for CPU
    return_dict=True,       # Return model outputs as a dictionary
    torch_dtype=torch.float16, # Use FP16 precision for computations
    device_map="auto",  # Map model to the appropriate device
)

# Load the fine-tuned model with LoRA weights and merge them into the base model
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()  # Merge LoRA weights and unload the merged model

# Reload the tokenizer to ensure compatibility with the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to EOS token
tokenizer.padding_side = "right"           # Set padding side to right

################################
# Upload Model to HuggingFace
################################
# Login to Hugging Face CLI
# huggingface-cli login --token <hf_...>  # USE CLI
model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)
