# --------- Install Required Packages ---------
# Install the Google Cloud AI Platform client library
# Note: Ensure you have Google Cloud SDK installed and configured
# pip install "google-cloud-aiplatform>=1.38"
# pip3 install --upgrade --user google-cloud-aiplatform

# Authenticate Google Cloud SDK (run this in your terminal)
# gcloud auth application-default login

# --------- Import Necessary Libraries ---------
import os
from pathlib import Path
import google.generativeai as genai

# --------- Configuration ---------
# Set up API key for Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define generation configuration parameters
generation_config = {
    "temperature": 0.4,  # Controls randomness in the output
    "top_p": 1,          # Controls diversity via nucleus sampling
    "top_k": 32,         # Limits the sampling pool to top_k tokens
    "max_output_tokens": 4096  # Maximum number of tokens in the output
}

# --------- Initialize the Model ---------
# Create an instance of the GenerativeModel with the specified configuration
model = genai.GenerativeModel(
    model_name="gemini-pro-vision",
    generation_config=generation_config
)

# --------- Generate Content from Image ---------
# Path to the image file
image_path = Path("image.jpeg")

# Load the image and prepare it for the model
image_part = {
    "mime_type": "image/jpeg",
    "data": image_path.read_bytes()
}

# Define the prompt and include the image part
prompt_parts = [
    "Describe what the people are doing in this image:\n",
    image_part
]

# Generate content based on the image and prompt
response = model.generate_content(prompt_parts)
print(response.text)

# -------------------------------------------------------------
from vertexai import generative_models
from vertexai.generative_models import GenerativeModel

# Initialize the GenerativeModel from Vertex AI
model = GenerativeModel(model_name="gemini-1.0-pro-vision")

# Generate content based on a question and image
response = model.generate_content(["What is this?", img])
print(response.text)

# -------------------------------------------------------------
import google.generativeai as genai
import os

# --------- Configuration ---------
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define generation configuration parameters
generation_config = {
    "temperature": 0.9,  # Controls randomness in the output
    "top_p": 1,          # Controls diversity via nucleus sampling
    "top_k": 1,          # Limits the sampling pool to top_k tokens
    "max_output_tokens": 2048  # Maximum number of tokens in the output
}

# --------- Initialize the Model ---------
# Create an instance of the GenerativeModel with the specified configuration
model = genai.GenerativeModel("gemini-pro", generation_config=generation_config)

# --------- Generate Content ---------
# Generate content based on a prompt
response = model.generate_content(["Create a Meal plan for today"])
print(response.text)

# --------------------------GRADIO UI--------------------------------
import gradio as gr
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part

# Define the function for generating content
def generate(prompt):
    model = GenerativeModel("gemini-pro")
    response = model.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": 2048,
            "temperature": 0.9,
            "top_p": 1
        },
        stream=False,
    )
    output = response.candidates[0].content.parts[0].text
    return output

# Set up the Gradio interface
iface = gr.Interface(
    fn=generate,                 # Function to call for generation
    inputs="text",               # Input type
    outputs="markdown",          # Output type
    title="Gemini Pro API UI"    # Interface title
)

# Launch the Gradio UI
iface.launch()
