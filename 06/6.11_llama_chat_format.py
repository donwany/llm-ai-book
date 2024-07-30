import random
import pandas as pd
from datasets import load_dataset

def format_to_llama2_chat(system_prompt, user_model_chat_list):
    growing_prompt = f"<s>[INST] <<SYS>> {system_prompt} <</SYS>>"
    for user_msg, model_answer in user_model_chat_list:
        growing_prompt += f" {user_msg} [/INST] {model_answer} </s>"
    return growing_prompt

def main():
    dataset_name = "nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions"
    # Load dataset
    legal_dataset = load_dataset(dataset_name)['train']
    data_list = []

    for sample in legal_dataset:
        instruction_input_separator = random.choice([":", ": ", "\n", "\n\n", " "])
        input_text = sample.get('input', '')
        instruction = sample.get('instruction', '')

        training_sequence = format_to_llama2_chat(
            "you are a helpful legal assistant",
            [(instruction + instruction_input_separator + input_text, sample['output'])]
        )
        data_list.append({"text": training_sequence})

    # Save the reformatted dataset locally
    pd.DataFrame(data_list).to_json("legal_dataset.jsonl", orient="records")

    # Load the reformatted dataset and push to the HuggingFace
    ds = load_dataset('json', data_files="legal_dataset.jsonl")
    ds.push_to_hub("worldboss/legal_chat", token="hf_xxx")

if __name__ == "__main__":
    main()
