# pip install datasets transformers torch evaluate nltk rouge_score
import evaluate

result = {"bleu": 0.7}
params = {"model": "gpt-2"}
if __name__ == '__main__':
    # Saves results to a JSON file
    evaluate.save("./results/", **result, **params)

    # Pushes the result of a metric to the metadata of a model repository in the Hub.
    evaluate.push_to_hub(
        model_id="huggingface/gpt2-wikitext2",
        metric_value=0.5,
        metric_type="bleu",
        metric_name="BLEU",
        dataset_name="WikiText",
        dataset_type="wikitext",
        dataset_split="test",
        task_type="text-generation",
        task_name="Text Generation"
    )
    # Logging methods
    evaluate.logging.set_verbosity_info()
