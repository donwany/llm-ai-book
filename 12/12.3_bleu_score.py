# pip install sacrebleu rouge_score datasets evaluate
import evaluate


def compute_bleu(predictions, references):
    # Load BLEU metric
    metric_bleu = evaluate.load("bleu")

    # Compute BLEU score
    bleu_score = metric_bleu.compute(predictions=predictions, references=references)

    return bleu_score


# Example predictions and references
predictions = ["hello there general kenobi", "foo bar foobar"]
references = [
    ["hello there general kenobi", "hello there !"],
    ["foo bar foobar"]
]

if __name__ == '__main__':
    # Compute BLEU score
    results = compute_bleu(predictions, references)
    print(results)
    # {'bleu': 1.0,
    # 'precisions': [1.0, 1.0, 1.0, 1.0],
    # 'brevity_penalty': 1.0,
    # 'length_ratio': 1.1666666666666667,
    # 'translation_length': 7,
    # 'reference_length': 7 }
