# --------- Setup ---------
# Install required packages
# pip install mlflow openai

# Import necessary libraries
import mlflow
import openai
import os
import pandas as pd

# --------- Data Preparation ---------
# Create a DataFrame with example questions and ground truth answers
eval_data = pd.DataFrame(
    {
        "inputs": [
            "What is Python?",
            "What is Kubernetes?",
        ],
        "ground_truth": [
            "Python is a high-level, interpreted programming language known for its readability and "
            "versatility. It is widely used for web development, data analysis, artificial intelligence, "
            "and scientific computing.",
            "Kubernetes is an open-source platform designed to automate the deployment, scaling, and "
            "operation of application containers. It was originally developed by Google and is now "
            "maintained by the Cloud Native Computing Foundation (CNCF).",
        ],
    }
)

# --------- MLflow Experiment Setup ---------
# Set up the MLflow experiment
mlflow.set_experiment("LLM Evaluation")

# Start a new MLflow run
with mlflow.start_run() as run:
    # Define the system prompt for the model
    system_prompt = "Answer the following question in two sentences"

    # Log the OpenAI GPT-4 model with MLflow
    logged_model_info = mlflow.openai.log_model(
        model="gpt-4",  # Specify the model name
        task=openai.chat.completions,  # Define the task for the model
        artifact_path="model",  # Path to save the model artifacts
        messages=[  # Define the message format
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "{question}"}
        ],
    )

    # Evaluate the model using predefined metrics
    results = mlflow.evaluate(
        logged_model_info.model_uri,  # URI of the logged model
        eval_data,  # Data for evaluation
        targets="ground_truth",  # Column in eval_data containing the ground truth answers
        model_type="question-answering",  # Specify the model type
        extra_metrics=[  # Additional metrics for evaluation
            mlflow.metrics.toxicity(),  # Evaluate toxicity
            mlflow.metrics.latency(),  # Evaluate latency
            mlflow.metrics.genai.answer_similarity()  # Evaluate answer similarity
        ]
    )

    # Retrieve and save the evaluation results
    eval_table = results.tables["eval_results_table"]
    df = pd.DataFrame(eval_table)
    df.to_csv('eval.csv')  # Save results to a CSV file
