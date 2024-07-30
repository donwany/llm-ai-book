# Fine-tune model via LLaMA Board
from llmtuner import create_ui

create_ui().queue().launch(share=True)
# Fine-tune model via Command Line
from llmtuner import run_exp

run_exp(dict(
    stage="sft", do_train=True,
    model_name_or_path="Qwen/Qwen1.5-0.5B-Chat",
    dataset="identity,alpaca_gpt4_en,alpaca_gpt4_zh",
    template="qwen",
    finetuning_type="lora",
    lora_target="all",
    output_dir="test_identity",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=100,
    learning_rate=1e-4,
    num_train_epochs=5.0,
    max_samples=500,
    max_grad_norm=1.0, fp16=True, ))

# Infer the fine-tuned model
from llmtuner import ChatModel

chat_model = ChatModel(dict(
    model_name_or_path="Qwen/Qwen1.5-0.5B-Chat",
    adapter_name_or_path="test_identity",
    finetuning_type="lora",
    template="qwen", ))
messages = []
while True:
    query = input("\nUser: ")
    if query.strip() == "exit":
        break
    if query.strip() == "clear":
        messages = []
        continue

    messages.append({"role": "user", "content": query})
    print("Assistant: ", end="", flush=True)
    response = ""
    for new_text in chat_model.stream_chat(messages):
        print(new_text, end="", flush=True)
        response += new_text
    print()
    messages.append({"role": "assistant", "content": response})

# Merge LoRA weights
from llmtuner import export_model

export_model(dict(
    model_name_or_path="Qwen/Qwen1.5-0.5B-Chat",
    adapter_name_or_path="test_identity",
    finetuning_type="lora",
    template="qwen",
    export_dir="test_exported",
    # export_hub_model_id="your_hf_id/test_identity",
))