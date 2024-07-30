from evaluate import load

glue_metric = load('glue', 'mrpc')
# sst2, mnli, mnli_mismatched, mnli_matched, qnli, rte, wnli, cola,stsb, mrpc, qqp, and hans
references = [0, 1]
predictions = [0, 1]
results = glue_metric.compute(predictions=predictions, references=references)

if __name__ == '__main__':
    print(results)  # {'accuracy': 1.0, 'f1': 1.0}
