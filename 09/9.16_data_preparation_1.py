from typing import Dict
from datasets import load_dataset
from transformers import DPOTrainer


def prepare_prompt_and_responses(samples) -> Dict[str, str]:
    """
    Prepare prompts and responses for the DPOTrainer.

    Args:
        samples (Dict[str, list]): A dictionary containing lists of questions and responses.

    Returns:
        Dict[str, list]: A dictionary with formatted prompts and responses.
    """
    return {
        "prompt": [
            f"Question: {question}\n\nAnswer: " for question in samples["question"]
        ],
        "chosen": samples["response_j"],  # Responses rated better than the alternative
        "rejected": samples["response_k"],  # Responses rated worse than the alternative
    }


# Load dataset
dataset = load_dataset(
    "lvwerra/stack-exchange-paired",
    split="train",
    data_dir="data/rl"
)

# Store original column names
original_columns = dataset.column_names

# Map function to format dataset
dataset = dataset.map(
    prepare_prompt_and_responses,  # Function to apply
    batched=True,  # Process the data in batches
    remove_columns=original_columns  # Remove old columns from the dataset
)
