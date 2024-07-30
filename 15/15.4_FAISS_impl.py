# Install the latest versions
# pip install -U langchain-community faiss-cpu langchain-openai tiktoken

import os
from getpass import getpass
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


def setup_environment():
    """Set up environment variables."""
    # Uncomment and set your OpenAI API Key if needed
    # os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API Key: ")
    # os.environ['FAISS_NO_AVX2'] = '1'


def load_and_process_documents(file_path):
    """Load and split documents into chunks."""
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)


def create_vector_store(docs):
    """Create a FAISS vector store from documents."""
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)


def perform_similarity_search(vector_store, query):
    """Perform a similarity search in the vector store."""
    return vector_store.similarity_search(query)


def main():
    setup_environment()

    # Load and process documents
    file_path = "../../modules/state_of_the_union.txt"
    docs = load_and_process_documents(file_path)

    # Create vector store
    vector_store = create_vector_store(docs)
    print(f"Total documents indexed: {vector_store.index.ntotal}")

    # Define query
    query = "What did the president say about Ketanji Brown Jackson"

    # Perform similarity search
    search_results = perform_similarity_search(vector_store, query)
    if search_results:
        print(search_results[0].page_content)
    else:
        print("No results found.")

    # Save and reload vector store
    vector_store.save_local("faiss_index")
    new_vector_store = FAISS.load_local("faiss_index", OpenAIEmbeddings())

    # Perform similarity search on reloaded vector store
    reloaded_search_results = perform_similarity_search(new_vector_store, query)
    if reloaded_search_results:
        print(reloaded_search_results[0].page_content)
    else:
        print("No results found after reloading.")


if __name__ == "__main__":
    main()
