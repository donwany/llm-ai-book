# Clone the EETQ repository and initialize submodules
# git clone https://github.com/NetEase-FuXi/EETQ.git
# cd EETQ/
# git submodule update --init --recursive
# pip install .

# Import the necessary modules
from eetq import AutoEETQForCausalLM
from transformers import AutoTokenizer

# Define the paths to your model and the directory where the quantized model will be saved
model_name = "/path/to/your/model"
quant_path = "/path/to/quantized/model"

# Load the tokenizer using the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model using the specified model name
model = AutoEETQForCausalLM.from_pretrained(model_name)

# Quantize the model and save the quantized version to the specified path
model.quantize(quant_path)

# Save the tokenizer to the same path as the quantized model
tokenizer.save_pretrained(quant_path)

# Load the quantized model in vllm and start the OpenAI API server with quantization enabled
# python -m vllm.entrypoints.openai.api_server \
#     --model /path/to/quantized/model  \
#     --quantization eetq \
#     --trust-remote-code

# git clone https://github.com/NetEase-FuXi/EETQ.git
# cd EETQ/
# git submodule update --init --recursive
# pip install .

# quantize the model
from eetq import AutoEETQForCausalLM
from transformers import AutoTokenizer

model_name = "/path/to/your/model"
quant_path = "/path/to/quantized/model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoEETQForCausalLM.from_pretrained(model_name)
model.quantize(quant_path)
tokenizer.save_pretrained(quant_path)

# Run the OpenAI API Server with Quantization:
# python -m vllm.entrypoints.openai.api_server \
#     --model /path/to/quantized/model \
#     --quantization eetq \
#     --trust-remote-code
