# Installation and setup
# git clone https://github.com/texttron/hyde.git
# pip install pyserini
# export OPENAI_API_KEY=<your_openai_key>  # Use your actual API key
# wget https://www.dropbox.com/s/dytqaqngaupp884/contriever_msmarco_index.tar.gz
# tar -xvf contriever_msmarco_index.tar.gz
# Demo: https://github.com/texttron/hyde/blob/main/hyde-demo.ipynb

# Import necessary libraries
import json
from pyserini.search import FaissSearcher, LuceneSearcher
from pyserini.search.faiss import AutoQueryEncoder
from hyde import Promptor, OpenAIGenerator, CohereGenerator, HyDE

# Define the API key for OpenAI or Cohere
API_KEY = '<your_api_key>'  # Replace with your OpenAI or Cohere API key

# Initialize components
promptor = Promptor('web search')
generator = OpenAIGenerator('text-davinci-003', API_KEY)
encoder = AutoQueryEncoder(encoder_dir='facebook/contriever', pooling='mean')
searcher = FaissSearcher('contriever_msmarco_index/', encoder)
corpus = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')

# Initialize HyDE
hyde = HyDE(promptor, generator, encoder, searcher)

# Define a query
query = 'how long does it take to remove wisdom tooth'

# Generate a prompt based on the query
prompt = hyde.prompt(query)
print("Generated Prompt:")
print(prompt)

# Generate hypothesis documents based on the query
print("\nGenerated Documents:")
hypothesis_documents = hyde.generate(query)
for i, doc in enumerate(hypothesis_documents):
    print(f'Document {i}:')
    print(doc.strip())

# Encode the query and hypothesis documents
hyde_vector = hyde.encode(query, hypothesis_documents)
print("\nEncoded Vector Shape:")
print(hyde_vector.shape)

# Perform a search using the encoded vector
print("\nSearch Results from Encoded Vector:")
hits = hyde.search(hyde_vector, k=10)
for i, hit in enumerate(hits):
    print(f'Retrieved Document {i}:')
    print(f'Document ID: {hit.docid}')
    print(json.loads(corpus.doc(hit.docid).raw())['contents'])

# Perform an end-to-end search with the query
print("\nEnd-to-End Search Results:")
hits = hyde.e2e_search(query, k=10)
for i, hit in enumerate(hits):
    print(f'Retrieved Document {i}:')
    print(f'Document ID: {hit.docid}')
    print(json.loads(corpus.doc(hit.docid).raw())['contents'])
