from transformers import AutoTokenizer

# Upload the tokenizer files to the 🤗 Model Hub.
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

if __name__ == '__main__':
    # Push the tokenizer to your namespace with the name "my-finetuned-bert".
    tokenizer.push_to_hub("my-finetuned-bert")

    # Push the tokenizer to an organization with the name "my-finetuned-bert".
    tokenizer.push_to_hub("huggingface/my-finetuned-bert")
