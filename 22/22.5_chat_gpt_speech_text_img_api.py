# ----------------
# SPEECH TO TEXT
# ----------------
from openai import OpenAI

client = OpenAI(api_key="sk-xxxxx")

audio_file = open("/path/to/file/audio.mp3", "rb")
transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file
)
print(transcription.text)

# --------------------
# TEXT TO SPEECH
# --------------------
client = OpenAI(api_key="sk-xxxxx")

speech_file_path = "speech.mp3"
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hi Everyone, This is Elon Musk"
)
response.stream_to_file(speech_file_path)

# -------------------------------
# DALLE-3 Image Generation API
# -------------------------------
client = OpenAI(api_key="sk-xxxxx")

response = client.images.generate(
    model="dall-e-3",
    prompt="a white siamese cat",
    size="1024x1024",
    quality="standard",
    n=1
)
image_url = response.data[0].url
print(image_url)
