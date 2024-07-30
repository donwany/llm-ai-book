# Usage
# Export Hugging Face token as an environment variable
# export HF_TOKEN=xxxxxxxxxxxxx

import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer
from datasets import load_dataset

def main():
    # Configurations
    max_seq_length = 2048
    url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
    dataset = load_dataset("json", data_files={"train": url}, split="train")

    # Load Mistral model
    model_name = "unsloth/mistral-7b-bnb-4bit"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Before training
    print("Before training\n")
    generate_text(model, tokenizer, "List the top 5 most popular movies of all time.")

    # Do model patching and add fast LoRA weights and training
    model = FastLanguageModel.get_peft_model(
        model, r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        max_seq_length=max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )

    # Training
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=60,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            output_dir="outputs",
            optim="adamw_8bit",
            seed=3407,
        ),
    )
    trainer.train()

    print("\n ######## \nAfter training\n")
    generate_text(model, tokenizer, "List the top 5 most popular movies of all time.")

    # Save and push to Hub
    model.save_pretrained("lora_model")
    model.save_pretrained_merged("outputs", tokenizer, save_method="merged_16bit")
    model.push_to_hub_merged("<HF_USERNAME>/mistral-7b-oig-unsloth-merged", tokenizer, save_method="merged_16bit", token=os.environ.get("HF_TOKEN"))
    model.push_to_hub("<HF_USERNAME>/mistral-7b-oig-unsloth", tokenizer, save_method="lora", token=os.environ.get("HF_TOKEN"))

def generate_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
