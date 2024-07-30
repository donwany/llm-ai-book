# pip install ctransformers
# git clone https://github.com/ggerganov/llama.cpp.git

# https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
# https://huggingface.co/TheBloke/LlamaGuard-7B-GGUF
# https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF

from ctransformers import AutoModelForCausalLM

# Define model path and file
model_path = "TheBloke/Mistral-7B-v0.1-GGUF"
model_file = "mistral-7b-v0.1.Q4_K_M.gguf"

# Set gpu_layers to the number of layers to offload to GPU, 0 for CPU only
gpu_layers = 50

# Load the model
llm = AutoModelForCausalLM.from_pretrained(model_path,
                                           model_file=model_file,
                                           model_type="mistral",
                                           gpu_layers=gpu_layers
                                           )

# Generate text with the model
output = llm("AI is going to")
print(output)

# https://github.com/ggerganov/llama.cpp/blob/master/convert-hf-to-gguf.py
