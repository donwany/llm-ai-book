from urllib.request import urlopen
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel

# Load an AI-generated image of a puppy playing in the snow
image = Image.open(urlopen("https://i.imgur.com/iQ5OtWi.png"))
caption = "a puppy playing in the snow"

model_id = "openai/clip-vit-base-patch32"

# Load a tokenizer to preprocess the text
tokenizer = CLIPTokenizerFast.from_pretrained(model_id)

# Load a processor to preprocess the images
processor = CLIPProcessor.from_pretrained(model_id)

# Main model for generating text and image embeddings
model = CLIPModel.from_pretrained(model_id)

# Tokenize our input
inputs = tokenizer(caption, return_tensors="pt")
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

# Create a text embedding
text_embedding = model.get_text_features(**inputs)
print(text_embedding.shape)

# Preprocess image
processed_image = processor(text=None, images=image, return_tensors='pt')['pixel_values']
print(processed_image.shape)

# Prepare image for visualization
img = np.einsum('ijk->jik', processed_image.squeeze(0).T)

# Visualize preprocessed image
plt.imshow(img)
plt.axis('off')
plt.show()

# Create the image embedding
image_embedding = model.get_image_features(processed_image)
print(image_embedding.shape)

# Calculate the probability of the text belonging to the image
text_probs = (100.0 * image_embedding @ text_embedding.T).softmax(dim=-1)
print(text_probs)

# Normalize the embeddings
text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

# Calculate their similarity
text_embedding = text_embedding.detach().cpu().numpy()
image_embedding = image_embedding.detach().cpu().numpy()
score = np.dot(text_embedding, image_embedding.T)
print(score)
