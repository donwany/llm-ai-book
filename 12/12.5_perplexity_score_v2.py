import evaluate

perplexity = evaluate.load("perplexity", module_type="metric")

input_texts = ["lorem ipsum",
               "Happy Birthday!",
               "Bienvenue",
               "ABC is a startup based in New York City and Paris"]

results = perplexity.compute(model_id='gpt2',
                             add_start_token=False,
                             predictions=input_texts)

if __name__ == '__main__':
    print(list(results.keys()))
    # ['perplexities', 'mean_perplexity']
    print(round(results["mean_perplexity"], 2))
    # 646.75
    print(round(results["perplexities"][0], 2))
    # 32.25
