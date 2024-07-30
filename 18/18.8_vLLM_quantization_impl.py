# pip install autoawq

# Quantize a model
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Define model paths and quantization configuration
model_path = 'lmsys/vicuna-7b-v1.5'
quant_path = 'vicuna-7b-v1.5-awq'
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# Load the model
model = AutoAWQForCausalLM.from_pretrained(model_path, **{"low_cpu_mem_usage": True})
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize the model
model.quantize(tokenizer, quant_config=quant_config)

# Save the quantized model and tokenizer
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

# Generate text with VLLM

from vllm import LLM, SamplingParams

# Sample prompts
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM instance with the quantized model
llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", quantization="AWQ")

# Generate text for each prompt
outputs = llm.generate(prompts, sampling_params)

# Print the results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
