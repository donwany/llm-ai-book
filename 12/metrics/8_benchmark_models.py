import pandas as pd
from datasets import load_dataset
from evaluate import evaluator
from transformers import pipeline
from evaluate.visualization import radar_plot

models = [
    "xlm-roberta-large-finetuned-conll03-english",
    "dbmdz/bert-large-cased-finetuned-conll03-english",
    "elastic/distilbert-base-uncased-finetuned-conll03-english",
    "dbmdz/electra-large-discriminator-finetuned-conll03-english",
    "gunghio/distilbert-base-multilingual-cased-finetuned-conll2003-ner",
    "philschmid/distilroberta-base-ner-conll2003",
    "Jorgeutd/albert-base-v2-finetuned-ner",
]

data = load_dataset("conll2003", split="validation").shuffle().select(1000)
task_evaluator = evaluator("token-classification")

results = []
for model in models:
    results.append(
        task_evaluator.compute(
            model_or_pipeline=model, data=data, metric="seqeval"
            )
        )

df = pd.DataFrame(results, index=models)
df[["overall_f1", "overall_accuracy", "total_time_in_seconds", "samples_per_second", "latency_in_seconds"]]

if __name__ == '__main__':
    plot = radar_plot(data=results, model_names=models, invert_range=["latency_in_seconds"])
    plot.show()