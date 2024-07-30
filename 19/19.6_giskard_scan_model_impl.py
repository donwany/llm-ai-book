# pip install "giskard[llm]" -U
# pip install langchain tiktoken "pypdf<=3.17.0"
import os

import giskard
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import PromptTemplate
from openai import OpenAI

# Set the OpenAI API Key environment variable
os.environ["OPENAI_API_KEY"] = "sk-..."

# Prepare vector store (FAISS) with IPCC report
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
loader = PyPDFLoader("https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf")
documents = loader.load_and_split(text_splitter)
db = FAISS.from_documents(documents, OpenAIEmbeddings())

# Prepare the QA chain
PROMPT_TEMPLATE = """You are the Climate Assistant, a helpful AI assistant made by Giskard.
Your task is to answer common questions on climate change.
You will be given a question and relevant excerpts from the IPCC Climate Change Synthesis Report (2023).
Please provide short and clear answers based on the provided context. Be polite and helpful.

Context:
{context}

Question:
{question}

Your answer:
"""

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question", "context"])
climate_qa_chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever(), prompt=prompt)


def model_predict(df: pd.DataFrame):
    """
    Wraps the LLM call in a simple Python function.

    This function takes a pandas DataFrame containing input variables needed by your model,
    and returns a list of outputs (one for each row).

    Args:
        df (pd.DataFrame): DataFrame containing input questions.

    Returns:
        list: List of responses from the model for each input question.
    """
    return [climate_qa_chain.run({"query": question}) for question in df["question"]]


# Define the Giskard model, including its name, description, and feature names
giskard_model = giskard.Model(
    model=model_predict,
    model_type="text_generation",
    name="Climate Change Question Answering",
    description="This model answers questions about climate change based on IPCC reports.",
    feature_names=["question"]
)

# Perform a scan of the model
scan_results = giskard.scan(giskard_model)

# Display the scan results
display(scan_results)

# Optionally, save the scan results to an HTML file
scan_results.to_html("scan_results.html")
