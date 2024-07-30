import torch, os, locale
import torchvision.transforms as transforms
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig

# 1. Setup: Initial configuration and model preparation
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = "HuggingFaceM4/idefics-9b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_skip_modules=["lm_head", "embed_tokens"]
)
processor = AutoProcessor.from_pretrained(checkpoint)
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, quantization_config=bnb_config, device_map="auto")
print(f"Model: {model}")


# 3. Image preprocessing: Convert image to RGB and apply transformations
def convert_to_rgb(image):
    if image.mode == "RGB":
        return image
    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


def ds_transforms(example_batch):
    image_size = processor.image_processor.image_size
    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std
    image_transform = transforms.Compose([
        convert_to_rgb,
        transforms.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std)
    ])
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


# 4. Dataset preparation: Loading and transforming the dataset for training
ds = load_dataset("julianmoraes/doodles-captions-BLIP")
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
eval_ds = ds["test"]
train_ds.set_transform(ds_transforms)
eval_ds.set_transform(ds_transforms)

# 5. Training: Setting up model training parameters and LoRA configuration
model_name = checkpoint.split("/")[1]
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
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
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds
)
trainer.train()

print(f"### After Fine Tuning ###\n")
image_path = "https://storage.googleapis.com/image-praison/wp-content/2024/03/b82ccbbd-image-5.jpg"
prompts = [
    [image_path, "Question: What's in this picture? Answer:"]
]


def do_inference(model, processor, prompts, max_new_tokens=50):
    tokenizer = processor.tokenizer
    bad_words = ["<image>", "<fake_token_around_image>"]
    if len(bad_words) > 0:
        bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids
    eos_token = "</s>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    inputs = processor(prompts, return_tensors="pt").to(device)
    generated_ids = model.generate(
        **inputs,
        eos_token_id=[eos_token_id],
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        num_beams=2,
        early_stopping=True
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)


do_inference(model, processor, prompts, max_new_tokens=50)

# 6. Model saving and pushing to HuggingFace Hub
locale.getpreferredencoding = lambda: "UTF-8"
model = model.merge_and_unload()
model.save_pretrained(f"{model_name}-doodles")
tokenizer = processor.tokenizer
tokenizer.save_pretrained(f"{model_name}-doodles")
model.push_to_hub(f"{model_name}-doodles", use_temp_dir=False, token=os.getenv("HF_TOKEN"))
tokenizer.push_to_hub(f"{model_name}-doodles", use_temp_dir=False, token=os.getenv("HF_TOKEN"))
