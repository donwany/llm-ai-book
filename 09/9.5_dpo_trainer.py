# imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer

# load model and dataset - dataset needs to be in a specific format
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# load trainer
trainer = DPOTrainer(model=model,
                     tokenizer=tokenizer, train_dataset=dataset,
                     )
# train
trainer.train()