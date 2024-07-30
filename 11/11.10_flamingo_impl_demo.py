# Install the OpenFlamingo package and its optional components
# pip install open-flamingo
# pip install open-flamingo[training]
# pip install open-flamingo[eval]
# pip install open-flamingo[all]

from open_flamingo import create_model_and_transforms
from PIL import Image
import requests
import torch

# --------- Initializing an OpenFlamingo model -------
model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",  # Path to the CLIP vision encoder model
    clip_vision_encoder_pretrained="openai",  # Pre-trained weights for the CLIP vision encoder
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",  # Path to the language model
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",  # Path to the tokenizer
    cross_attn_every_n_layers=1,  # Frequency of cross-attention layers
    cache_dir="PATH/TO/CACHE/DIR"  # Directory for caching models (defaults to ~/.cache)
)

# --------- Generating Text ---------
"""
Step 1: Load images
"""
# Load images from URLs
demo_image_one = Image.open(
    requests.get(
        "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
    ).raw
)
demo_image_two = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
        stream=True
    ).raw
)
query_image = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028352.jpg",
        stream=True
    ).raw
)

"""
Step 2: Preprocessing images
Details: OpenFlamingo expects images as torch tensors of shape 
batch_size x num_media x num_frames x channels x height x width. 
Here batch_size = 1, num_media = 3 (three images), num_frames = 1,
channels = 3 (RGB), height = 224, width = 224.
"""
# Preprocess the images
vision_x = [
    image_processor(demo_image_one).unsqueeze(0),
    image_processor(demo_image_two).unsqueeze(0),
    image_processor(query_image).unsqueeze(0)
]
# Stack images into a single tensor
vision_x = torch.cat(vision_x, dim=0)
# Add batch and media dimensions
vision_x = vision_x.unsqueeze(1).unsqueeze(0)

"""
Step 3: Preprocessing text
Details: The text should include an <image> special token to indicate where an image is,
and an <end> special token to indicate the end of the text associated with an image.
"""
# Set tokenizer padding side to the left for generation
tokenizer.padding_side = "left"
# Tokenize the text with special tokens for image references
lang_x = tokenizer(
    ["<image>An image of two cats.<image>An image of a bathroom sink.<image>An image of"],
    return_tensors="pt"
)

"""
Step 4: Generate text
"""
# Generate text using the model
generated_text = model.generate(
    vision_x=vision_x,  # Input images
    lang_x=lang_x["input_ids"],  # Tokenized text input
    attention_mask=lang_x["attention_mask"],  # Attention mask for the text input
    max_new_tokens=20,  # Maximum number of new tokens to generate
    num_beams=3  # Number of beams for beam search
)
# Decode and print the generated text
print("Generated text: ", tokenizer.decode(generated_text[0]))
