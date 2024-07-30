# imports
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer

# load model and dataset - dataset needs to be in a specific format
model = AutoModelForSequenceClassification.from_pretrained(
    "gpt2",
    num_labels=1
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# load trainer
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
)
# train
trainer.train()