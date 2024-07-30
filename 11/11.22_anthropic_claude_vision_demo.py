# --------- Installation ---------
# Install the `anthropic` package
# pip install anthropic

# --------- Imports ---------
import base64  # For encoding images to base64
import anthropic  # For interacting with Anthropic's API

# --------- Initialize Client ---------
# Create an Anthropic client instance with your API key
client = anthropic.Client(api_key="YOUR_API_KEY")

# Define the model to use
MODEL_NAME = "claude-3-opus-20240229"


# --------- Helper Function ---------
def get_base64_encoded_image(image_path):
    """
    Convert an image file to a base64-encoded string.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()  # Read the image file in binary mode
        base_64_encoded_data = base64.b64encode(binary_data)  # Encode binary data to base64
        base64_string = base_64_encoded_data.decode('utf-8')  # Convert bytes to string
        return base64_string


# --------- Transcribing Handwritten Text ---------
# Create a list of messages for the API request
message_list = [
    {
        "role": 'user',  # Role of the message sender
        "content": [
            # Image content encoded in base64
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": get_base64_encoded_image("../images/transcribe/school_notes.png")}},
            # Text instruction
            {"type": "text", "text": "Transcribe this text. Only output the text and nothing else."}
        ]
    }
]

# Send the request to the Anthropic API
response = client.messages.create(
    model=MODEL_NAME,  # Model to use for the request
    max_tokens=2048,  # Maximum number of tokens to generate
    messages=message_list  # Messages to send to the model
)

# Print the transcription result
print(response.content[0].text)

# --------- Converting Unstructured Information to JSON ---------
# Create a list of messages for the API request
message_list = [
    {
        "role": 'user',  # Role of the message sender
        "content": [
            # Image content encoded in base64
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": get_base64_encoded_image("../images/transcribe/org_chart.jpeg")}},
            # Text instruction
            {"type": "text",
             "text": "Turn this org chart into JSON indicating who reports to who. Only output the JSON and nothing else."}
        ]
    }
]

# Send the request to the Anthropic API
response = client.messages.create(
    model=MODEL_NAME,  # Model to use for the request
    max_tokens=2048,  # Maximum number of tokens to generate
    messages=message_list  # Messages to send to the model
)

# Print the JSON result
print(response.content[0].text)
