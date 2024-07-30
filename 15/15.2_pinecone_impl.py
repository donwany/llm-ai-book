# Install the latest version of Pinecone client
# pip3 install pinecone-client

import os
import pinecone


def initialize_pinecone_client():
    """Initialize the Pinecone client."""
    api_key = os.getenv("PINECONE_API_KEY", "your-api-key")
    env = os.getenv("PINECONE_ENVIRONMENT", "your-environment")
    pinecone.init(api_key=api_key, environment=env)
    return pinecone


def create_index(pc, index_name, dimension, metric, environment, pod_type):
    """Create a Pinecone index."""
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=PodSpec(environment=environment, pod_type=pod_type)
    )


def list_indexes(pc):
    """List all Pinecone indexes."""
    indexes = pc.list_indexes()
    for index in indexes:
        print(index['name'])


def describe_index(pc, index_name):
    """Describe a specific Pinecone index."""
    return pc.describe_index(index_name)


def delete_index(pc, index_name):
    """Delete a specific Pinecone index."""
    pc.delete_index(index_name)


def configure_index(pc, index_name, replicas):
    """Configure the number of replicas for an index."""
    pc.configure_index(index_name, replicas=replicas)


def describe_index_stats(pc, index_host):
    """Describe index statistics."""
    index = pc.Index(host=index_host)
    return index.describe_index_stats()


def upsert_vectors(index, namespace, vectors, metadata=None):
    """Upsert vectors into a Pinecone index."""
    return index.upsert(vectors=vectors, namespace=namespace, metadata=metadata)


def query_vectors(index, namespace, vector, top_k, include_values=True, include_metadata=True, filter=None):
    """Query vectors in a Pinecone index."""
    return index.query(
        namespace=namespace,
        vector=vector,
        top_k=top_k,
        include_values=include_values,
        include_metadata=include_metadata,
        filter=filter
    )


def delete_vectors(index, ids, namespace):
    """Delete vectors from a Pinecone index."""
    return index.delete(ids=ids, namespace=namespace)


def fetch_vectors(index, ids, namespace):
    """Fetch vectors from a Pinecone index."""
    return index.fetch(ids=ids, namespace=namespace)


def update_vector(index, id, values, set_metadata, namespace):
    """Update a vector in a Pinecone index."""
    return index.update(id=id, values=values, set_metadata=set_metadata, namespace=namespace)


def create_collection(pc, collection_name, source_index):
    """Create a Pinecone collection."""
    pc.create_collection(name=collection_name, source=source_index)


def list_collections(pc):
    """List all Pinecone collections."""
    return pc.list_collections()


def describe_collection(pc, collection_name):
    """Describe a specific Pinecone collection."""
    return pc.describe_collection(collection_name)


def delete_collection(pc, collection_name):
    """Delete a specific Pinecone collection."""
    pc.delete_collection(collection_name)


# Main execution
if __name__ == "__main__":
    pc = initialize_pinecone_client()

    # Example operations
    create_index(pc, "example-index", 1536, "cosine", 'us-west-2', 'p1.x1')
    list_indexes(pc)

    index_description = describe_index(pc, "example-index")
    print("Index Description:", index_description)

    delete_index(pc, "example-index")

    configure_index(pc, "example-index", replicas=4)

    index_stats = describe_index_stats(pc, os.environ.get('INDEX_HOST'))
    print("Index Stats:", index_stats)

    index = pc.Index(host=os.environ.get('INDEX_HOST'))

    upsert_response = upsert_vectors(index, "example-namespace", [{"id": "vec1", "values": [0.1, 0.2, 0.3, 0.4]}])
    print("Upsert Response:", upsert_response)

    query_response = query_vectors(index, "example-namespace", [0.1, 0.2, 0.3, 0.4], top_k=10,
                                   filter={"genre": {"$in": ["comedy", "documentary", "drama"]}})
    print("Query Response:", query_response)

    delete_response = delete_vectors(index, ["vec1", "vec2"], "example-namespace")
    print("Delete Response:", delete_response)

    fetch_response = fetch_vectors(index, ["vec1", "vec2"], "example-namespace")
    print("Fetch Response:", fetch_response)

    update_response = update_vector(index, "vec1", [0.1, 0.2, 0.3, 0.4], {"genre": "drama"}, "example-namespace")
    print("Update Response:", update_response)

    create_collection(pc, "example-collection", "example-index")
    collections = list_collections(pc)
    print("Collections:", collections)

    collection_description = describe_collection(pc, "example-collection")
    print("Collection Description:", collection_description)

    delete_collection(pc, "example-collection")
