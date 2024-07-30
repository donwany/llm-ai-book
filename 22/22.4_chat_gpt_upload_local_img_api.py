import base64
import requests

# OpenAI API Key
API_KEY = "sk-xxxxx"


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def main():
    # Path to your image
    image_path = "path_to_your_image.jpg"
    # Getting the base64 string
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {"role": "user",
             "content": [
                 {
                     "type": "text",
                     "text": "What's in this image?"
                 },
                 {
                     "type": "image_url",
                     "image_url": {
                         "url": f"data:image/jpeg;base64,{base64_image}"
                     }
                 }
             ]
             }
        ],
        "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(response.json())


if __name__ == "__main__":
    main()
