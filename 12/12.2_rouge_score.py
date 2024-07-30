# pip install sacrebleu rouge_score datasets evaluate
import evaluate

def compute_rouge(predictions, references):
    # Load ROUGE metric
    metric_rouge = evaluate.load("rouge")

    # Compute ROUGE score
    rouge_score = metric_rouge.compute(predictions=predictions, references=references)

    return rouge_score


# Example predictions and references
predictions = ["hello there", "general kenobi"]
references = ["hello there", "general kenobi"]

if __name__ == '__main__':
    # Compute ROUGE score
    results = compute_rouge(predictions, references)
    print(results)  # {'rouge1': 1.0, 'rouge2': 1.0, 'rougeL': 1.0, 'rougeLsum': 1.0}
