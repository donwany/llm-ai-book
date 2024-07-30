# pip install huggingface_hub

from datasets import load_dataset

dataset = load_dataset("filesFolder", data_dir="/path/to/pokemon")

if __name__ == '__main__':
    # push-to-hub
    dataset = load_dataset("stevhliu/demo")
    dataset.push_to_hub("stevhliu/processed_demo")

    # push as private
    dataset.push_to_hub("stevhliu/private_processed_demo", private=True)

    # load .csv
    data_files = {"train": "train.csv", "test": "test.csv"}
    dataset = load_dataset("namespace/your_dataset_name", data_files=data_files)
    dataset.push_to_hub("namespace/processed_demo")
    # load parquet file
    dataset = load_dataset("parquet", data_files={'train': 'train.parquet', 'test': 'test.parquet'})
    dataset.push_to_hub("namespace/processed_demo")
    # load as arrow file
    dataset = load_dataset("arrow", data_files={'train': 'train.arrow', 'test': 'test.arrow'})
    # load text files
    dataset = load_dataset("text", data_files={"train": ["my_text_1.txt", "my_text_2.txt"],
                                               "test": "my_test_file.txt"})
    dataset = load_dataset("text", data_dir="path/to/text/dataset")
    # load zipped csv files
    url = "https://domain.org/train_data.zip"
    data_files = {"train": url}
    dataset = load_dataset("csv", data_files=data_files)