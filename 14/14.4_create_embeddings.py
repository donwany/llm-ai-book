# Install the cohere library
# pip install cohere

import cohere
import numpy as np

# Initialize Cohere client with your API key
api_key = "YOUR_API_KEY"
co = cohere.Client(api_key)

# Define phrases to get embeddings for
phrases = ["i love soup", "soup is my favorite", "london is far away"]

# Specify model and input type
model = "embed-english-v3.0"
input_type = "search_query"

# Retrieve embeddings for the given phrases
response = co.embed(
    texts=phrases,
    model=model,
    input_type=input_type,
    embedding_types=['float']
)

# Extract embeddings for each phrase
soup1, soup2, london = response.embeddings.float


# Define a function to calculate cosine similarity between two vectors
def calculate_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity between vec1 and vec2.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Compute and print similarity between different phrases
similarity_soup = calculate_similarity(soup1, soup2)
similarity_london = calculate_similarity(soup1, london)

print(f"Similarity between 'i love soup' and 'soup is my favorite': {similarity_soup:.2f}")
print(f"Similarity between 'i love soup' and 'london is far away': {similarity_london:.2f}")
