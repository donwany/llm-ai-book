# conda create -n finetuning python=3.11 -y
# conda activate finetuning
# export HF_HUB_ENABLE_HF_TRANSFER=1
# export HF_TOKEN=xxxxxxxxxx
# pip install datasets transformers bitsandbytes sentencepiece accelerate loralib peft pillow torch torchvision hf_transfer

# --------- Import Necessary Libraries ---------
import torch
import os
import locale
import torchvision.transforms as transforms
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig

# --------- 1. Setup: Initial Configuration and Model Preparation ---------
# Define the device to use (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Specify the model checkpoint
checkpoint = "HuggingFaceM4/idefics-9b"

# Configure BitsAndBytes for quantization to optimize memory usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_skip_modules=["lm_head", "embed_tokens"]
)

# Load the processor and model from Hugging Face
processor = AutoProcessor.from_pretrained(checkpoint)
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, quantization_config=bnb_config, device_map="auto")

print(f"Model: {model}")


# --------- 2. Before Fine-Tuning: Inference Function ---------
def do_inference(model, processor, prompts, max_new_tokens=50):
    """
    Generate text based on image prompts using the model.
    """
    tokenizer = processor.tokenizer
    bad_words = ["<image>", "<fake_token_around_image>"]
    bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids if len(bad_words) > 0 else []
    eos_token = "</s>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)

    # Prepare input tensors
    inputs = processor(prompts, return_tensors="pt").to(device)

    # Generate text from the model
    generated_ids = model.generate(
        **inputs,
        eos_token_id=[eos_token_id],
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        num_beams=2,
        early_stopping=True
    )

    # Decode and print the generated text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)


# Test inference before fine-tuning
print(f"### Before Fine Tuning ###\n")
image_path = "https://storage.googleapis.com/image-praison/wp-content/2024/03/b82ccbbd-image-5.jpg"
prompts = [[image_path, "Question: What's in this picture? Answer:"]]
do_inference(model, processor, prompts, max_new_tokens=50)


# --------- 3. Image Preprocessing: Convert Image to RGB and Apply Transformations ---------
def convert_to_rgb(image):
    """
    Convert an image to RGB format if it is not already in RGB.
    """
    if image.mode == "RGB":
        return image
    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


def ds_transforms(example_batch):
    """
    Apply transformations to the dataset images and prepare inputs for the model.
    """
    image_size = processor.image_processor.image_size
    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std

    # Define image transformation pipeline
    image_transform = transforms.Compose([
        convert_to_rgb,
        transforms.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std)
    ])

    # Prepare prompts and inputs for the model
    prompts = []
    for i in range(len(example_batch['text'])):
        prompts.append(
            [
                example_batch['image'][i],
                f"Question: What's in this picture? Answer: This is {example_batch['text'][i]}.</s>"
            ],
        )

    inputs = processor(prompts, transform=image_transform, return_tensors="pt").to(device)
    inputs["labels"] = inputs["input_ids"]
    return inputs


# --------- 4. Dataset Preparation: Load and Transform Dataset for Training ---------
# Load the dataset and split into training and evaluation sets
ds = load_dataset("julianmoraes/doodles-captions-BLIP")
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
eval_ds = ds["test"]

# Apply transformations to the dataset
train_ds.set_transform(ds_transforms)
eval_ds.set_transform(ds_transforms)

# --------- 5. Training: Set Up Model Training Parameters and LoRA Configuration ---------
# Define model name
model_name = checkpoint.split("/")[1]

# Configure LoRA (Low-Rank Adaptation) for efficient training
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

# Apply LoRA to the model
model = get_peft_model(model, config)
model.print_trainable_parameters()

# Define training arguments
training_args = TrainingArguments(
    output_dir=f"{model_name}-logo",
    learning_rate=2e-4,
    fp16=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    dataloader_pin_memory=False,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=10,
    save_steps=25,
    max_steps=25,
    logging_steps=5,
    remove_unused_columns=False,
    push_to_hub=False,
    label_names=["labels"],
    load_best_model_at_end=False,
    report_to="none",
    optim="paged_adamw_8bit",
)

# Create Trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds
)
trainer.train()

# Test inference after fine-tuning
print(f"### After Fine Tuning ###\n")
image_path = "https://storage.googleapis.com/image-praison/wp-content/2024/03/b82ccbbd-image-5.jpg"
prompts = [[image_path, "Question: What's in this picture? Answer:"]]
do_inference(model, processor, prompts, max_new_tokens=50)

# --------- 6. Model Saving and Pushing to Hugging Face Hub ---------
# Ensure UTF-8 encoding for locale
locale.getpreferredencoding = lambda: "UTF-8"

# Merge and unload model, save locally, and push to Hugging Face Hub
model = model.merge_and_unload()
model.save_pretrained(f"{model_name}-doodles")
tokenizer = processor.tokenizer
tokenizer.save_pretrained(f"{model_name}-doodles")
model.push_to_hub(f"{model_name}-doodles", use_temp_dir=False, token=os.getenv("HF_TOKEN"))
tokenizer.push_to_hub(f"{model_name}-doodles", use_temp_dir=False, token=os.getenv("HF_TOKEN"))
