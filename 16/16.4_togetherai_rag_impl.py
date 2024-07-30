# Install the necessary packages (uncomment to use)
# pip install --quiet pypdf chromadb tiktoken openai langchain-together

# Load PDF document
from langchain_community.document_loaders import PyPDFLoader

pdf_path = "mixtral.pdf"
loader = PyPDFLoader(pdf_path)
data = loader.load()

# Split document into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Add chunks to vector database
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Uncomment to use TogetherEmbeddings instead
# from langchain_together.embeddings import TogetherEmbeddings
# embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)

retriever = vectorstore.as_retriever()
