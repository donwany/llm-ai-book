# Install the required version
# python3 -m pip install pymilvus==2.0.x

from pymilvus import connections, Collection


def connect_to_milvus(host: str, port: str, alias: str = "default"):
    """Connect to the Milvus server."""
    connections.connect(alias=alias, host=host, port=port)


def create_collection(collection_name: str, schema, shards_num: int = 2, consistency_level: str = "Strong"):
    """Create a collection with the specified schema."""
    return Collection(name=collection_name, schema=schema,
                      using='default', shards_num=shards_num, consistency_level=consistency_level)


def build_index(collection: Collection, field_name: str, index_params):
    """Build an index for the specified field."""
    collection.create_index(field_name=field_name, index_params=index_params)


def insert_data(collection: Collection, data):
    """Insert data into the collection."""
    return collection.insert(data)


def vector_search(collection: Collection, query_vector: list, anns_field: str, search_params, limit: int = 10):
    """Perform a vector search in the collection."""
    return collection.search(
        data=query_vector,
        anns_field=anns_field,
        param=search_params,
        limit=limit,
        expr=None,
        consistency_level="Strong"
    )


def vector_query(collection: Collection, expr: str, output_fields: list):
    """Perform a vector query in the collection."""
    return collection.query(
        expr=expr,
        output_fields=output_fields,
        consistency_level="Strong"
    )


def main():
    # Configuration
    host = 'localhost'
    port = '19530'
    collection_name = "book"

    # Connect to Milvus
    connect_to_milvus(host, port)

    # Define schema, index_params, and search_params (example values)
    schema = ...  # Define the schema according to your data
    index_params = ...  # Define index parameters
    search_params = ...  # Define search parameters

    # Create a collection
    collection = create_collection(collection_name, schema)

    # Build an index
    build_index(collection, field_name="book_intro", index_params=index_params)

    # Insert data
    data = ...  # Define the data to be inserted
    insert_data(collection, data)

    # Perform vector search
    query_vector = [[0.1, 0.2]]  # Example query vector
    search_results = vector_search(collection, query_vector, anns_field="book_intro", search_params=search_params)
    print("Search results:", search_results)

    # Perform vector query
    query_expr = "book_id in [2,4,6,8]"
    query_results = vector_query(collection, expr=query_expr, output_fields=["book_id", "book_intro"])
    print("Query results:", query_results)


if __name__ == "__main__":
    main()
