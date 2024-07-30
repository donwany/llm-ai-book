import openai  # Import the OpenAI library
import cv2  # Import the OpenCV library for video processing
import base64  # Import base64 library for encoding frames


def get_frames(video_path):
    """Extract frames from a video file and encode them in base64 format."""
    video = cv2.VideoCapture(video_path)  # Open the video file
    frames = []  # Initialize an empty list to store frames
    while video.isOpened():  # Loop until the video is open
        success, frame = video.read()  # Read a frame from the video
        if not success:  # Break the loop if no frame is read
            break
        _, buffer = cv2.imencode(".jpg", frame)  # Encode the frame as JPEG
        frames.append(base64.b64encode(buffer).decode("utf-8"))  # Convert to base64 and add to the list
    video.release()  # Release the video capture object
    return frames  # Return the list of base64 encoded frames


def main():
    """Main function to extract video frames and interact with OpenAI API."""
    client = openai.OpenAI(api_key="sk-xxxxx")  # Initialize OpenAI client with API key
    video_frames = get_frames("video.mp4")  # Extract frames from the video file
    # Create a completion request with the first 5 frames to the GPT-4 Vision model
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{"role": "user", "content": [{"image": frame} for frame in video_frames[:5]]}]
    )
    print(response.choices[0].message.content)  # Print the response from the model


if __name__ == "__main__":
    main()  # Run the main function if the script is executed directly
