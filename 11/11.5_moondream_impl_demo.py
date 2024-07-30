# Install necessary packages
# pip install transformers einops

# Clone the repository and navigate into the directory
# git clone https://github.com/vikhyat/moondream.git
# cd moondream
# pip install -r requirements.txt

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch  # Import torch for tensor operations

# Define the model ID and revision
model_id = "vikhyatk/moondream2"
revision = "2024-05-20"

# Load the model with specified settings
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,  # Allow the use of code from the model's repository
    revision=revision,       # Use the specified revision of the model
    torch_dtype=torch.float16,  # Use float16 precision for performance
    attn_implementation="flash_attention_2"  # Use the flash attention implementation for efficiency
).to("cuda")  # Move the model to GPU

# Load the tokenizer associated with the model
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# Open an image and encode it using the model
image = Image.open('<IMAGE_PATH>')  # Replace <IMAGE_PATH> with the path to your image
enc_image = model.encode_image(image)  # Encode the image into a format the model can process

# Generate a response from the model based on a question about the image
print(model.answer_question(enc_image, "Describe this image.", tokenizer))

# Batch inference: process multiple images with different prompts
answers = moondream.batch_answer(
    images=[
        Image.open('<IMAGE_PATH_1>'),  # Replace <IMAGE_PATH_1> with the path to the first image
        Image.open('<IMAGE_PATH_2>')   # Replace <IMAGE_PATH_2> with the path to the second image
    ],
    prompts=[
        "Describe this image.",  # Prompt for the first image
        "Are there people in this image?"  # Prompt for the second image
    ],
    tokenizer=tokenizer  # Use the same tokenizer for both images
)
