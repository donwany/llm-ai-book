# Upgrade the diffusers package to include torch support
# pip install --upgrade diffusers[torch]

from diffusers import DiffusionPipeline
import torch

# Load the Stable Diffusion pipeline with pre-trained weights and set precision to float16
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)

# Move the pipeline to the GPU for faster processing
pipeline.to("cuda")

# Generate an image based on the text prompt and retrieve the first image from the results
image = pipeline("An image of a squirrel in Picasso style").images[0]
image
# ----------------------------------------------------------------

from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import torch

# Load the DDPM scheduler and UNet2D model with pre-trained weights
scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")

# Set the number of timesteps for the diffusion process
scheduler.set_timesteps(50)

# Get the sample size from the model configuration
sample_size = model.config.sample_size

# Create a random noise tensor as the initial input
noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")
input = noise

# Perform the diffusion process to generate an image
for t in scheduler.timesteps:
    with torch.no_grad():
        # Compute the noisy residual and the previous noisy sample
        noisy_residual = model(input, t).sample
        prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
        input = prev_noisy_sample

# Convert the final image tensor to a format suitable for viewing
image = (input / 2 + 0.5).clamp(0, 1)  # Rescale and clamp pixel values to [0, 1]
image = image.cpu().permute(0, 2, 3, 1).numpy()[0]  # Move to CPU and convert to NumPy array
image = Image.fromarray((image * 255).round().astype("uint8"))  # Convert to PIL Image
image  # Display the image
