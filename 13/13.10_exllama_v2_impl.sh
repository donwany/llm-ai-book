#!/bin/bash

# Clone the exllamav2 repository and navigate into the directory
# git clone https://github.com/turboderp/exllamav2
# cd exllamav2

# Optionally, create and activate a new conda environment

# Install the required dependencies and the exllamav2 package
# pip install -r requirements.txt
# pip install .
# pip install exllamav2

# Convert a model and create a directory containing the quantized version
python convert.py \
    -i /mnt/models/llama2-7b-fp16/ \
    -o /mnt/temp/exl2/ \
    -nr \
    -m /mnt/models/llama2-7b-exl2/measurement.json \
    -cf /mnt/models/llama2-7b-exl2/4.0bpw/ \
    -b 4.0

python convert.py \
    -i /mnt/models/llama2-7b-fp16/ \
    -o /mnt/temp/exl2/ \
    -nr \
    -m /mnt/models/llama2-7b-exl2/measurement.json \
    -cf /mnt/models/llama2-7b-exl2/4.5bpw/ \
    -b 4.5

# Parameters:
# -i: Path to the base model to convert (in HF format, FP16).
# -o: Path to the working directory for temporary files and final output.
# -m: Path to the measurement file.
# -cf: Path to the calibration file.
# -b: Target average number of bits per weight (bpw). For example, 4.0 bpw stores weights in 4-bit precision.

# Perform inference
python exllamav2/test_inference.py -m quant/ -p "I have a dream"


# Another Example here:
# https://colab.research.google.com/drive/1yrq4XBlxiA0fALtMoT2dwiACVc77PHou?usp=sharing

# EXL2 Model:
# https://huggingface.co/mlabonne/zephyr-7b-beta-5.0bpw-exl2