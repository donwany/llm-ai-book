# Clone the LLaVA repository and set up the environment
# git clone https://github.com/haotian-liu/LLaVA.git
# cd LLaVA
# conda create -n llava python=3.10 -y
# conda activate llava
# pip install --upgrade pip  # Upgrade pip to enable PEP 660 support
# pip install -e .  # Install the LLaVA package in editable mode

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# Define the path to the pre-trained model
model_path = "liuhaotian/llava-v1.5-7b"

# Load the pre-trained model, tokenizer, image processor, and context length
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,                   # Path to the model
    model_base=None,                        # Base model (if any)
    model_name=get_model_name_from_path(model_path)  # Extract the model name from the path
)

# Define the query prompt and image file for evaluation
prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"

# Create a placeholder for arguments to pass to the evaluation function
args = type('Args', (), {
    "model_path": model_path,               # Path to the model
    "model_base": None,                    # Base model (if any)
    "model_name": get_model_name_from_path(model_path),  # Extract the model name
    "query": prompt,                       # User query prompt
    "conv_mode": None,                    # Conversation mode (if any)
    "image_file": image_file,             # URL of the image file
    "sep": ",",                          # Separator for multiple choices (if any)
    "temperature": 0,                     # Temperature for sampling (0 for deterministic)
    "top_p": None,                       # Top-p (nucleus) sampling parameter (if any)
    "num_beams": 1,                      # Number of beams for beam search
    "max_new_tokens": 512                # Maximum number of new tokens to generate
})()

# Run the evaluation with the defined arguments
eval_model(args)

# CLI Inference Example
# Use the command-line interface for inference with LLaVA
# python -m llava.serve.cli \
#     --model-path liuhaotian/llava-v1.5-7b \  # Path to the model
#     --image-file "https://llava-vl.github.io/static/images/view.jpg" \  # Image file for inference
#     --load-4bit  # Option to load the model with 4-bit quantization
