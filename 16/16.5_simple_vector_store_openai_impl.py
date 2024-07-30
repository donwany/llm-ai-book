# Install the necessary packages (uncomment to use)
# pip install llama-index-core llama-index-llms-openai

import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Import required modules
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents from the specified directory
data_directory = "YOUR_DATA_DIRECTORY"
documents = SimpleDirectoryReader(data_directory).load_data()

# Create an index from the loaded documents
index = VectorStoreIndex.from_documents(documents)
