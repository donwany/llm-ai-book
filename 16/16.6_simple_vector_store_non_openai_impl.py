# Install the necessary package (uncomment to use)
# pip install llama-index

import os
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.replicate import Replicate
from transformers import AutoTokenizer

# Set environment variable for Replicate API token
os.environ["REPLICATE_API_TOKEN"] = "YOUR_REPLICATE_API_TOKEN"

# Configure LLM settings
llama2_7b_chat_model = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
Settings.llm = Replicate(
    model=llama2_7b_chat_model,
    temperature=0.01,
    additional_kwargs={"top_p": 1, "max_new_tokens": 300},
)

# Configure tokenizer to match the LLM
Settings.tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")

# Configure embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Load documents from the specified directory
data_directory = "YOUR_DATA_DIRECTORY"
documents = SimpleDirectoryReader(data_directory).load_data()

# Create an index from the loaded documents
index = VectorStoreIndex.from_documents(documents)

# Create a query engine from the index and run a query
query_engine = index.as_query_engine()
query_result = query_engine.query("YOUR_QUESTION")
print(query_result)

# Persist the index data in memory
index.storage_context.persist()
