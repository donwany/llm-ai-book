# pip install sacrebleu rouge_score datasets evaluate
import evaluate

seqeval = evaluate.load('seqeval')
predictions = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
references = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]

results = seqeval.compute(predictions=predictions, references=references)
print(results)
# {'MISC': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 1},
# 'PER': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 1},
# 'overall_precision': 1.0, 'overall_recall': 1.0, 'overall_f1': 1.0, 'overall_accuracy': 1.0}

