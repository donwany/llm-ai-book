# Clone the Stable Diffusion repository and set up the environment
# git clone https://github.com/CompVis/stable-diffusion.git
# cd scripts

# Install PyTorch and torchvision for deep learning
# conda install pytorch torchvision -c pytorch

# Install required Python packages
# pip install transformers==4.19.2 diffusers invisible-watermark
# pip install -e .  # Install the Stable Diffusion package in editable mode

# Generate an image from text using the txt2img script
# python scripts/txt2img.py \
#        --prompt "a photograph of an astronaut riding a horse" \  # Text prompt for image generation
#        --plms  # Use PLMS sampling for image generation
#
# Modify an existing image based on a new prompt using the img2img script
# python scripts/img2img.py \
#     --prompt "A fantasy landscape, trending on artstation" \  # Text prompt to guide the image modification
#     --init-img <path-to-img.jpg> \  # Path to the initial image to be modified
#     --strength 0.8  # Strength of the modification effect (0 to 1)

# Ensure you are logged in with Hugging Face CLI for model access
# huggingface-cli login

from torch import autocast
from diffusers import StableDiffusionPipeline

# Load the pre-trained Stable Diffusion model from Hugging Face
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",  # Model ID for Stable Diffusion
    use_auth_token=True  # Use authentication token for accessing the model
).to("cuda")  # Move the model to GPU for faster processing

# Define the text prompt for image generation
prompt = "a photo of an astronaut riding a horse on mars"

# Generate the image with autocasting for mixed precision
with autocast("cuda"):
    image = pipe(prompt)["sample"][0]  # Generate the image and extract the sample

# Save the generated image to a file
image.save("astronaut_rides_horse.png")
