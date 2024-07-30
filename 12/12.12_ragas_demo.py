# --------- Setup ---------
# Install the required package
# pip install ragas
# Clone the repository and install
# git clone https://github.com/explodinggradients/ragas && cd ragas
# pip install -e .

# --------- Import Libraries ---------
from datasets import Dataset
import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness

# --------- Set Environment Variables ---------
# Set your OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# --------- Prepare Data ---------
# Create a dictionary with sample data
data_samples = {
    'question': [
        'When was the first Super Bowl?',
        'Who won the most Super Bowls?'
    ],
    'answer': [
        'The first Super Bowl was held on Jan 15, 1967',
        'The most Super Bowls have been won by The New England Patriots'
    ],
    'contexts': [
        ['The First AFL-NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles'],
        ['The Green Bay Packers...Green Bay, Wisconsin.','The Packers compete...Football Conference']
    ],
    'ground_truth': [
        'The first Super Bowl was held on January 15, 1967',
        'The New England Patriots have won the Super Bowl a record six times'
    ]
}

# Convert the dictionary to a Dataset object
dataset = Dataset.from_dict(data_samples)

# --------- Evaluate Metrics ---------
# Evaluate the dataset using specified metrics
score = evaluate(dataset, metrics=[faithfulness, answer_correctness])

# Convert the evaluation score to a pandas DataFrame and display it
score.to_pandas()
