from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt

# --------- Define an Inference Source ---------
# Path to the image file you want to run inference on
source = 'path/to/bus.jpg'

# --------- Create a FastSAM Model ---------
# Initialize the FastSAM model with a pre-trained weight file.
# You can choose either 'FastSAM-s.pt' or 'FastSAM-x.pt' based on your needs.
model = FastSAM('https://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-s.pt')

# --------- Run Inference on an Image ---------
# Perform inference on the image using the FastSAM model.
# Parameters:
# - device: 'cpu' specifies to use the CPU for computation.
# - retina_masks: Whether to use high-resolution masks.
# - imgsz: Size of the input image (e.g., 1024 pixels).
# - conf: Confidence threshold for detection.
# - iou: Intersection over Union threshold for Non-Maximum Suppression.
everything_results = model(source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

# --------- Prepare a Prompt Process Object ---------
# Initialize the FastSAMPrompt object with the source image and inference results.
prompt_process = FastSAMPrompt(source, everything_results, device='cpu')

# --------- Generate Annotations ---------
# Everything Prompt: Get annotations for all detected objects.
ann = prompt_process.everything_prompt()

# Bbox Prompt: Get annotations for a specific bounding box.
# Bounding box format: [x1, y1, x2, y2]
ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])

# Text Prompt: Get annotations based on a text description.
ann = prompt_process.text_prompt(text='a photo of a dog')

# Point Prompt: Get annotations based on specific points.
# Points format: [[x1, y1], [x2, y2], ...]
# Point labels: 0 for background, 1 for foreground
ann = prompt_process.point_prompt(points=[[200, 200]], pointlabel=[1])

# --------- Plot Annotations ---------
# Plot the annotations on the image and save the output.
# The output directory where the annotated image will be saved.
prompt_process.plot(annotations=ann, output='./')
