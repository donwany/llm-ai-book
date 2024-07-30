#!/bin/bash

# --------- Setup ---------
# Clone the G-Eval repository from GitHub
git clone https://github.com/nlpyang/geval.git

# Navigate into the geval directory
cd geval

# --------- Fluency Evaluation ---------
# Evaluate fluency on the SummEval dataset
# Replace <sk-xxxxx...> with your OpenAI API key
python gpt4_eval.py \
    --prompt prompts/summeval/flu_detailed.txt \  # Path to the prompt file for fluency evaluation
    --save_fp results/gpt4_flu_detailed.json \  # Path to save the evaluation results
    --summeval_fp data/summeval.json \  # Path to the SummEval dataset
    --key <sk-xxxxx...>  # Your OpenAI API key

# --------- Meta Evaluation ---------
# Meta-evaluate the G-Eval results
# Specify the path to the G-Eval results and the evaluation dimension
python meta_eval_summeval.py \
    --input_fp results/gpt4_flu_detailed.json \  # Path to the G-Eval results for fluency
    --dimension fluency  # Dimension to evaluate (in this case, fluency)
