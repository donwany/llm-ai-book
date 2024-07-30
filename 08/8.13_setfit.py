# Install SetFit
# pip install setfit
# OR
# pip install git+https://github.com/huggingface/setfit.git

from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset

# Load a dataset from the Hugging Face Hub
dataset = load_dataset("sst2")

# Simulate the few-shot regime by sampling 8 examples per class
train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=8)
eval_dataset = dataset["validation"].select(range(100))
test_dataset = dataset["validation"].select(range(100, len(dataset["validation"])))

# Load a SetFit model from Hub
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2", labels=["negative", "positive"])

args = TrainingArguments(
    batch_size=16,
    num_epochs=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    metric="accuracy",
    column_mapping={"sentence": "text", "label": "label"}  # Map dataset columns to text/label expected by trainer
)

# Train and evaluate
trainer.train()
metrics = trainer.evaluate(test_dataset)
print(metrics)  # {'accuracy': 0.8691709844559585}

# Push model to the Hub
trainer.push_to_hub("tomaarsen/setfit-paraphrase-mpnet-base-v2-sst2")

# Download from Hub
model = SetFitModel.from_pretrained("tomaarsen/setfit-paraphrase-mpnet-base-v2-sst2")

# Run inference
preds = model.predict(["i loved the spiderman movie!", "pineapple on pizza is the worst"])
print(preds)  # ["positive", "negative"]
