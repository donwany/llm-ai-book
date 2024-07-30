# pip install ultralytics
from ultralytics import SAM

# Load a model
model = SAM('https://github.com/ultralytics/assets/releases/download/v8.2.0/sam_b.pt')
# Display model information (optional)
model.info()
# Run inference with bboxes prompt
model('ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709])
# Run inference with points prompt
model('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1])

# from ultralytics import SAM
# # Load a model
# model = SAM('https://github.com/ultralytics/assets/releases/download/v8.2.0/sam_l.pt')
# # Display model information (optional)
# model.info()
# # Run inference
# model('path/to/image.jpg')

# from ultralytics import SAM
# # Load the model
# model = SAM('https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt')
# # Predict a segment based on a point prompt
# model.predict('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1])
