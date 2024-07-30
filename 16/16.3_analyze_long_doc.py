# Install the necessary packages (uncomment to use)
# pip install --upgrade langchain langchain-google-vertexai

# Read the document
document_path = "../docs/docs/modules/state_of_the_union.txt"
with open(document_path, 'r') as file:
    state_of_the_union = file.read()

# Import the necessary modules
from langchain.chains import AnalyzeDocumentChain
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Initialize the OpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Load the QA chain
qa_chain = load_qa_chain(llm, chain_type="map_reduce")

# Initialize the AnalyzeDocumentChain
qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

# Run the QA document chain
question = "What did the president say about Justice Breyer?"
answer = qa_document_chain.run(
    input_document=state_of_the_union,
    question=question
)

print(answer)
