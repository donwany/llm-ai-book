# Install the latest version
# pip install qdrant-client

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


def create_qdrant_client(memory_mode: bool, path: str = None) -> QdrantClient:
    """Create and return a Qdrant client."""
    if memory_mode:
        return QdrantClient(":memory:")  # In-memory Qdrant instance for testing
    return QdrantClient(path=path)  # Persistent storage on disk


def create_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """Create or recreate a collection with specified vector parameters."""
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )


def insert_vectors(client: QdrantClient, collection_name: str, vectors: np.ndarray):
    """Insert vectors into the specified collection."""
    points = [
        PointStruct(
            id=idx,
            vector=vector.tolist(),
            payload={"color": "red", "rand_number": idx % 10}
        )
        for idx, vector in enumerate(vectors)
    ]
    client.upsert(collection_name=collection_name, points=points)


def search_vectors(client: QdrantClient, collection_name: str, query_vector: np.ndarray, limit: int):
    """Search for vectors in the specified collection."""
    return client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit
    )


def main():
    # Initialize client
    qdrant_client = create_qdrant_client(
        memory_mode=True)  # Change to `memory_mode=False` and provide path for persistent storage

    # Create collection
    collection_name = "my_collection"
    vector_size = 100
    create_collection(qdrant_client, collection_name, vector_size)

    # Insert vectors
    vectors = np.random.rand(100, vector_size)
    insert_vectors(qdrant_client, collection_name, vectors)

    # Search vectors
    query_vector = np.random.rand(vector_size)
    hits = search_vectors(qdrant_client, collection_name, query_vector, limit=5)

    print(hits)


if __name__ == "__main__":
    main()
