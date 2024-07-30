# pip install llama-cpp-python

from llama_cpp import Llama

# Initialize the LLaMA model
llm = Llama(model_path="./models/7B/llama-model.gguf",
            # n_gpu_layers=-1, # Uncomment to use GPU acceleration
            # seed=1337, # Uncomment to set a specific seed
            # n_ctx=2048, # Uncomment to increase the context window
            )

# Generate a completion
output = llm("Q: Name the planets in the solar system? A: ",  # Prompt
             max_tokens=32,  # Generate up to 32 tokens, set to None to generate up to the end of the context window
             stop=["Q:", "\n"],  # Stop generating just before the model would generate a new question
             echo=True  # Echo the prompt back in the output
             )

# Print the result
print(output)
# {
#   "id": "cmpl-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
#   "object": "text_completion",
#   "created": 1679561337,
#   "model": "./models/7B/llama-model.gguf",
#   "choices": [
#     {
#       "text": "Q: Name the planets in the solar system? A: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune and Pluto.",
#       "index": 0,
#       "logprobs": None,
#       "finish_reason": "stop"
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 14,
#     "completion_tokens": 28,
#     "total_tokens": 42
#   }
# }

# Load and Use a model from huggingface hub
# Pulling models from Hugging Face Hub
llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
    filename="*q8_0.gguf",
    verbose=False
)

# Repo: https://github.com/abetlen/llama-cpp-python.git
