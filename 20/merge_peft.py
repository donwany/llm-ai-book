# Example usage:
# python merge_peft.py --base_model=meta-llama/Llama-2-7b-hf --peft_model=./qlora-out --hub_id=alpaca-qlora

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--peft_model", type=str)
    parser.add_argument("--hub_id", type=str)

    return parser.parse_args()


def main():
    args = get_args()

    print(f"[1/5] Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"[2/5] Loading adapter: {args.peft_model}")
    model = PeftModel.from_pretrained(base_model, args.peft_model, device_map="auto")

    print("[3/5] Merge base model and adapter")
    model = model.merge_and_unload()

    print(f"[4/5] Saving model and tokenizer in {args.hub_id}")
    model.save_pretrained(f"{args.hub_id}")
    tokenizer.save_pretrained(f"{args.hub_id}")

    print(f"[5/5] Uploading to Hugging Face Hub: {args.hub_id}")
    model.push_to_hub(f"{args.hub_id}", use_temp_dir=False)
    tokenizer.push_to_hub(f"{args.hub_id}", use_temp_dir=False)

    print("Merged model uploaded to Hugging Face Hub!")


if __name__ == "__main__":
    main()