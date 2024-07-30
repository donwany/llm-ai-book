from datasets import list_metrics

metrics_list = list_metrics()

if __name__ == '__main__':
    len(metrics_list)
    print(metrics_list)
    # ['accuracy',
    # 'bertscore', 'bleu',
    # 'bleurt', 'cer',
    # 'comet', 'coval',
    # 'cuad', 'f1',
    # 'gleu', 'glue',
    # 'indic_glue',
    # 'matthews_correlation', 'meteor',
    # 'pearsonr', 'precision',
    # 'recall', 'rouge',
    # 'sacrebleu', 'sari',
    # 'seqeval', 'spearmanr',
    # 'squad', 'squad_v2',
    # 'super_glue', 'wer',
    # 'wiki_split', 'xnli']
