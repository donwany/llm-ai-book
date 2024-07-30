# --------- Install AutoAWQ ---------
# Install the autoawq package
# pip install autoawq

# Clone the AutoAWQ repository from GitHub
# git clone https://github.com/casper-hansen/AutoAWQ
# cd AutoAWQ

# Install the AutoAWQ package in editable mode
# pip install -e .

# --------- Quantization --------------
# Import necessary modules
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Define model and quantization paths
model_path = 'lmsys/vicuna-7b-v1.5'
quant_path = 'vicuna-7b-v1.5-awq'

# Define quantization configuration
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# Load the base model
model = AutoAWQForCausalLM.from_pretrained(model_path)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Perform quantization
model.quantize(tokenizer, quant_config=quant_config)

# Save the quantized model and tokenizer
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)


# --------- Inference ----------------------
# Import necessary modules for inference
from transformers import AutoTokenizer, TextStreamer

# Define the path to the quantized model
quant_path = "TheBloke/zephyr-7B-beta-AWQ"

# Load the quantized model
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)

# Initialize the text streamer for output
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Define the prompt template
prompt_template = """<|system|></s><|user|>{prompt}</s><|assistant|>"""

# Define the prompt
prompt = ("You're standing on the surface of the Earth. "
          "You walk one mile south, one mile west and one mile north. "
          "You end up exactly where you started. Where are you?")

# Convert the prompt to tokens
tokens = tokenizer(prompt_template.format(prompt=prompt), return_tensors='pt').input_ids.cuda()

# Generate the output using the quantized model
generation_output = model.generate(tokens, streamer=streamer, max_seq_len=512)
