# pip install sacrebleu rouge_score datasets evaluate
from evaluate.visualization import radar_plot

data = [
    {"accuracy": 0.99, "precision": 0.8, "f1": 0.95, "latency_in_seconds": 33.6},
    {"accuracy": 0.98, "precision": 0.87, "f1": 0.91, "latency_in_seconds": 11.2},
    {"accuracy": 0.98, "precision": 0.78, "f1": 0.88, "latency_in_seconds": 87.6},
    {"accuracy": 0.88, "precision": 0.78, "f1": 0.81, "latency_in_seconds": 101.6}
]

model_names = ["Llama-7B", "Gemma-7B", "Mistral-7B", "Falcon-7B"]

if __name__ == '__main__':
    plot = radar_plot(data=data, model_names=model_names)
    plot.show()
