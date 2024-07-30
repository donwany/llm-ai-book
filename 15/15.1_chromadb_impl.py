# Install the necessary library
# pip install chromadb

import chromadb


def setup_chroma_client():
    """Set up a Chroma client for in-memory storage."""
    # Initialize the Chroma client
    client = chromadb.Client()
    return client


def create_and_populate_collection(client):
    """Create a collection, add documents to it, and perform a search query."""
    # Create a collection named "all-my-documents"
    collection = client.create_collection("all-my-documents")

    # Add documents to the collection with metadata and unique IDs
    collection.add(
        documents=[
            "This is document1",
            "This is document2"
        ],
        metadatas=[
            {"source": "notion"},
            {"source": "google-docs"}
        ],
        ids=[
            "doc1",
            "doc2"
        ]
    )

    return collection


def query_collection(collection):
    """Query the collection and return the most similar results."""
    # Perform a search query to find the 2 most similar results
    results = collection.query(
        query_texts=["This is a query document"],
        n_results=2
        # Optional filters can be added here if needed
        # where={"metadata_field": "is_equal_to_this"},
        # where_document={"$contains": "search_string"}
    )

    return results


# Main execution
if __name__ == "__main__":
    client = setup_chroma_client()
    collection = create_and_populate_collection(client)
    search_results = query_collection(collection)

    # Print search results
    print("Search Results:", search_results)
