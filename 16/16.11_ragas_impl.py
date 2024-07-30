# Install the necessary package (uncomment to use)
# pip install ragas

# Clone the repository and install the package (uncomment to use)
# git clone https://github.com/explodinggradients/ragas
# cd ragas
# pip install -e .

from datasets import Dataset
import os
from ragas import evaluate
from ragas.metrics import (answer_relevancy, faithfulness,
                           answer_correctness, context_recall,
                           context_precision)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# Define sample data
data_samples = {
    'question': [
        'When was the first Super Bowl?',
        'Who won the most Super Bowls?'
    ],
    'answer': [
        'The first Super Bowl was held on Jan 15, 1967',
        'The most Super Bowls have been won by The New England Patriots'
    ],
    "contexts": [
        ["The First AFL-NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,"],
        ["The Green Bay Packers...Green Bay, Wisconsin.","The Packers compete...Football Conference"]
    ],
    "ground_truth": [
        "The first Super Bowl was held on January 15, 1967",
        "The New England Patriots have won the Super Bowl a record six times"
    ]
}

# Create a dataset from the sample data
dataset = Dataset.from_dict(data_samples)

# Evaluate the dataset using specified metrics
score = evaluate(dataset, metrics=[faithfulness, answer_correctness])

# Convert the score to a pandas DataFrame and display
score_df = score.to_pandas()
print(score_df)
