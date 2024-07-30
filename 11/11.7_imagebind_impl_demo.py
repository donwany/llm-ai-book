# Create a new conda environment named 'imagebind' with Python 3.8
# conda create --name imagebind python=3.8 -y

# Activate the newly created conda environment
# conda activate imagebind

# Install the current package (assuming a local setup)
# pip install .

# Install the 'soundfile' package for audio processing
# pip install soundfile

from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

# Define lists of inputs for different modalities
text_list = ["A dog.", "A car", "A bird"]
image_paths = [".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
audio_paths = [".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]

# Set the device to CUDA if a GPU is available, otherwise use CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate the ImageBind model with pre-trained weights
model = imagebind_model.imagebind_huge(pretrained=True)

# Set the model to evaluation mode and move it to the appropriate device
model.eval()
model.to(device)

# Load and preprocess data for each modality
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

# Perform inference and compute embeddings
with torch.no_grad():
    embeddings = model(inputs)

# Compute and print similarity scores between different modalities
print(
    "Vision x Text: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
)

print(
    "Audio x Text: ",
    torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
)

print(
    "Vision x Audio: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
)
