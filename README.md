var = [
    {
        "prompt": "Once upon a time, in a faraway land, there lived a brave knight.",
        "response": "The brave knight embarked on a quest to rescue the captured princess."
    },
    {
        "prompt": "In a galaxy far, far away...",
        "response": "The rebels launched a daring attack on the Death Star."
    }
]


[
  {
    "context": "The quick brown fox jumps over the lazy dog.",
    "question": "What color is the fox?",
    "answer": "brown"
  },
  {
    "context": "Albert Einstein was a theoretical physicist who developed the theory of relativity.",
    "question": "What is Albert Einstein known for?",
    "answer": "developing the theory of relativity"
  }
]

[
  {
    "text": "I loved the movie. The acting was amazing!",
    "label": "positive"
  },
  {
    "text": "The service was terrible. I will never go back.",
    "label": "negative"
  }
]

[
  {
    "sentence1": "The cat is sleeping on the mat.",
    "sentence2": "The cat is awake.",
    "label": "contradiction"
  },
  {
    "sentence1": "The sun rises in the east.",
    "sentence2": "The sun sets in the west.",
    "label": "entailment"
  }
]

[
  {
    "document": "Scientists have discovered a new species of plant in the Amazon rainforest. The plant has unique purple flowers and grows in areas with high humidity.",
    "summary": "A new species of purple-flowered plant has been found in the Amazon rainforest."
  },
  {
    "document": "The company reported a decline in profits for the third quarter. Sales were lower than expected due to weak consumer demand.",
    "summary": "The company's profits dropped in the third quarter due to weak consumer demand."
  }
]

[
  {
    "source_language": "english",
    "target_language": "spanish",
    "source_text": "Hello, how are you?",
    "target_text": "Hola, ¿cómo estás?"
  },
  {
    "source_language": "french",
    "target_language": "english",
    "source_text": "Le chat noir",
    "target_text": "The black cat"
  }
]

[
  {
    "history": ["Hi, how are you?", "I'm good, thanks! How about you?"],
    "response": "I'm doing well too."
  },
  {
    "history": ["What's your favorite food?", "Pizza! What about you?"],
    "response": "I love sushi."
  }
]

{"chosen": "<prompt + good response>", "rejected": "<prompt + worse response>"}
{"chosen": "<prompt + good response>", "rejected": "<prompt + worse response>"}
{"chosen": "<prompt + good response>", "rejected": "<prompt + worse response>"}
