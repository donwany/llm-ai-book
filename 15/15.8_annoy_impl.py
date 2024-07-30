# Install the latest version of annoy
# pip install --upgrade --quiet annoy

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Annoy


def create_vector_store(texts, embeddings_func, metric="angular", n_trees=100, n_jobs=-1):
    """
    Create an Annoy VectorStore from the given texts.

    Parameters:
    - texts: List of text strings to be embedded.
    - embeddings_func: Function or object for generating embeddings.
    - metric: Distance metric to use for Annoy (default is "angular").
    - n_trees: Number of trees to use for Annoy index (default is 100).
    - n_jobs: Number of parallel jobs to use for Annoy index creation (default is -1).

    Returns:
    - vector_store: An Annoy VectorStore instance.
    """
    return Annoy.from_texts(texts, embeddings_func, metric=metric, n_trees=n_trees, n_jobs=n_jobs)


def perform_similarity_search(vector_store, query, k=3):
    """
    Perform a similarity search on the vector store.

    Parameters:
    - vector_store: An Annoy VectorStore instance.
    - query: The query text for which to find similar texts.
    - k: Number of nearest neighbors to return (default is 3).

    Returns:
    - search_results: List of texts similar to the query.
    """
    return vector_store.similarity_search(query, k=k)


def perform_similarity_search_with_score(vector_store, query, k=3):
    """
    Perform a similarity search with scores on the vector store.

    Parameters:
    - vector_store: An Annoy VectorStore instance.
    - query: The query text for which to find similar texts.
    - k: Number of nearest neighbors to return (default is 3).

    Returns:
    - search_results_with_scores: List of tuples (text, score) similar to the query.
    """
    return vector_store.similarity_search_with_score(query, k=k)


def save_vector_store(vector_store, file_path):
    """
    Save the vector store to a local file.

    Parameters:
    - vector_store: An Annoy VectorStore instance.
    - file_path: Path where the vector store will be saved.
    """
    vector_store.save_local(file_path)


def main():
    # Create embeddings function
    embeddings_func = HuggingFaceEmbeddings()

    # Define texts
    texts = ["pizza is great", "I love salad", "my car", "a dog"]

    # Create vector stores with different parameters
    vector_store_default = create_vector_store(texts, embeddings_func)
    vector_store_custom = create_vector_store(texts, embeddings_func, metric="dot", n_trees=100, n_jobs=1)

    # Perform similarity searches
    search_results_default = perform_similarity_search(vector_store_default, "food", k=3)
    search_results_with_scores = perform_similarity_search_with_score(vector_store_default, "food", k=3)

    # Print results
    print("Default Search Results:", search_results_default)
    print("Search Results with Scores:", search_results_with_scores)

    # Save the vector store
    save_vector_store(vector_store_default, "my_annoy_index_and_docstore")


if __name__ == "__main__":
    main()
