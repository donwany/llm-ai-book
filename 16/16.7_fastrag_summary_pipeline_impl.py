from haystack.document_stores import InMemoryDocumentStore
from haystack.schema import Document
from haystack.nodes import BM25Retriever, SentenceTransformersRanker
from haystack.nodes.prompt import PromptNode, PromptTemplate, AnswerParser
from haystack.agents.memory import ConversationSummaryMemory
from haystack import Pipeline

# Build a local in-memory index and index some relevant documents
document_store = InMemoryDocumentStore(use_gpu=False, use_bm25=True)

# Example documents to index
examples = [
    "Lionel Andres Messi ...",
    "Born and raised in central ...",
    "An Argentine international, ...",
    "Messi has endorsed sportswear",
]
documents = [Document(content=d, id=i) for i, d in enumerate(examples)]

# Write documents to the document store
document_store.write_documents(documents)

# Initialize the pipeline components
retriever = BM25Retriever(document_store=document_store, top_k=100)
reranker = SentenceTransformersRanker(
    model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2",
    top_k=10
)

# Configure the prompt template and reader
prompt_template = PromptTemplate(
    prompt="{join(documents)}",
    output_parser=AnswerParser()
)
reader = PromptNode(
    "philschmid/bart-large-cnn-samsum",
    model_kwargs={"task_name": "text2text-generation"},
    default_prompt_template=prompt_template
)

# Create and configure the pipeline
pipeline = Pipeline()
pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
pipeline.add_node(component=reranker, name="Reranker", inputs=["Retriever"])
pipeline.add_node(component=reader, name="Reader", inputs=["Reranker"])

# Run a query through the pipeline
query_result = pipeline.run(query="Leo Messi")

# Display the summary
print(query_result['answers'][0].answer)
