# Import necessary libraries
import numpy as np
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if CUDA (GPU) is available; otherwise, use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the tokenizer with the pre-trained GPT-2 model
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load the pre-trained GPT-2 model for causal language modeling and move it to the specified device
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

# Loop over different values of `use_cache` (True and False)
for use_cache in (True, False):
    times = []
    # Measure the generation time for 10 iterations
    for _ in range(10):
        # Record the start time
        start = time.time()

        # Generate text based on input prompt, specifying device and use_cache
        model.generate(
            **tokenizer("What is KV caching?", return_tensors="pt").to(device),
            use_cache=use_cache,
            max_new_tokens=1000
        )

        # Record the time taken for generation
        times.append(time.time() - start)

    # Calculate and print average generation time with standard deviation for each `use_cache` setting
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"{'with' if use_cache else 'without'} KV caching: {round(avg_time, 3)} ± {round(std_time, 3)} seconds")
