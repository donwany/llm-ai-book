# Install the latest version of Weaviate client
# pip install -U weaviate-client

import os
import json
import requests
import weaviate
import weaviate.classes as wvc


def connect_to_weaviate():
    """Connect to Weaviate instance."""
    cluster_url = os.getenv("WCS_CLUSTER_URL")
    api_key = os.getenv("WCS_API_KEY")
    openai_api_key = os.getenv("OPENAI_APIKEY")

    client = weaviate.Client(
        url=cluster_url,
        auth_client_secret=weaviate.AuthApiKey(api_key),
        additional_headers={"X-OpenAI-Api-Key": openai_api_key}
    )

    return client


def create_collection(client):
    """Create a new collection in Weaviate."""
    collection_config = wvc.config.Configure(
        vectorizer=wvc.config.Vectorizer.text2vec_openai(),
        generative=wvc.config.Generative.openai()
    )

    client.schema.create_class(
        {
            "class": "Question",
            "vectorizer": collection_config.vectorizer,
            "generative": collection_config.generative,
            "properties": [
                {"name": "answer", "dataType": ["text"]},
                {"name": "question", "dataType": ["text"]},
                {"name": "category", "dataType": ["text"]}
            ]
        }
    )


def add_objects_to_collection(client):
    """Add objects to the 'Question' collection."""
    response = requests.get(
        'https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json')
    data = response.json()  # Load data

    question_objs = [
        {
            "answer": d["Answer"],
            "question": d["Question"],
            "category": d["Category"]
        }
        for d in data
    ]

    client.batch.add_objects(question_objs, class_name="Question")
    client.batch.flush()  # Ensure all objects are added


def query_collection(client):
    """Query the 'Question' collection."""
    results = client.query.get(class_name="Question", properties=["answer", "question", "category"]) \
        .with_near_text({"concepts": ["biology"]}) \
        .with_limit(2) \
        .do()

    if results.get('objects'):
        for obj in results['objects']:
            print(obj['properties'])  # Inspect the properties of the objects


def main():
    client = None
    try:
        client = connect_to_weaviate()
        create_collection(client)
        add_objects_to_collection(client)
        query_collection(client)
    finally:
        if client:
            client.close()  # Close client gracefully


if __name__ == "__main__":
    main()
