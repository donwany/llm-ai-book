# Install necessary packages using conda and pip
# conda install --yes -c pytorch=1.7.1 torchvision cudatoolkit=11.0
# pip install ftfy regex tqdm
# pip install git+https://github.com/openai/CLIP.git

import torch
import clip
from PIL import Image

# Determine the device to use: CUDA if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and preprocessing function
# 'ViT-B/32' specifies the Vision Transformer model with 32x32 patch size
model, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess the image
# 'CLIP.png' is the image file to be processed
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)

# Tokenize the text prompts
# These are the text descriptions to compare against the image
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# Perform inference
with torch.no_grad():  # Disable gradient calculations for inference
    # Encode the image and text into feature vectors
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # Compute similarity logits between image and text features
    logits_per_image, logits_per_text = model(image, text)

    # Convert logits to probabilities using softmax
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# Print the probabilities of each label for the given image
print("Label probs:", probs)
# Expected output format: [[prob_a, prob_b, prob_c]], where each probability corresponds to one of the text prompts
