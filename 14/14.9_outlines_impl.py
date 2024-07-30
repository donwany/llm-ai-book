# Install the necessary library
# pip install outlines

import outlines


# Define a simple addition function
def add(a: int, b: int) -> int:
    return a + b


# ------------------- Using the Outlines Library for JSON Generation -------------------
def generate_json_response():
    # Load the transformer model for JSON generation
    model = outlines.models.transformers("WizardLM/WizardMath-7B-V1.1")

    # Create a JSON generator using the model and the add function
    generator = outlines.generate.json(model, add)

    # Generate a JSON with the required integers
    result = generator("Return json with two integers named a and b respectively. a is odd and b even.")

    # Compute the result using the add function and print it
    print("Addition Result:", add(**result))  # Expected output: 3


# ------------------- Using the Outlines Library for Multiple Choices -------------------
def generate_sentiment_analysis():
    # Load the transformer model for choice generation
    model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")

    # Define the prompt for sentiment analysis
    prompt = """You are a sentiment-labelling assistant.
    Is the following review positive or negative?

    Review: This restaurant is just awesome!
    """

    # Create a choice generator using the model and the possible choices
    generator = outlines.generate.choice(model, ["Positive", "Negative"])

    # Generate the sentiment analysis result
    answer = generator(prompt)

    # Print the sentiment analysis result
    print("Sentiment Analysis Result:", answer)


# Call functions to execute the respective tasks
generate_json_response()
generate_sentiment_analysis()
