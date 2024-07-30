import requests
import random

url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
response = requests.get(url)
alpaca = response.json()
random_row = random.choice(alpaca)

def prompt_input(row):
    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ).format_map(row)

print(prompt_input(random_row))
