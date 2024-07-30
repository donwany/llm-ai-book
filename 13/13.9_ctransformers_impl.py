# Install the ctransformers package and optional dependencies
# pip install ctransformers
# pip install ctransformers[cuda]
# pip install ctransformers[gptq]

from ctransformers import AutoModelForCausalLM

# Load a GGML model
llm = AutoModelForCausalLM.from_pretrained("/path/to/ggml-model.bin", model_type="gpt2")
print(llm("AI is going to"))

# Streaming response
for text in llm("AI is going to", stream=True):
    print(text, end="", flush=True)

# Load a model from a repository with multiple model files
llm = AutoModelForCausalLM.from_pretrained("marella/gpt-2-ggml", model_file="ggml-model.bin")

# Use the Transformers text generation pipeline
from transformers import pipeline, AutoTokenizer

# Load model and tokenizer for the pipeline
tokenizer = AutoTokenizer.from_pretrained("gpt2")
pipe = pipeline("text-generation", model=llm, tokenizer=tokenizer)

# Generate text
print(pipe("AI is going to", max_new_tokens=256))
pipe("AI is going to", max_new_tokens=256, do_sample=True, temperature=0.8, repetition_penalty=1.1)

# Load a GPTQ model
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-GPTQ")
print(llm("AI is going to"))

# Colab Notebooks:
# https://colab.research.google.com/drive/1FVSLfTJ2iBbQ1oU2Rqz0MkpJbaB_5Got
# https://colab.research.google.com/drive/1SzHslJ4CiycMOgrppqecj4VYCWFnyrN0
