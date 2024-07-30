# pip install langgraph langchain_openai
from transformers import load_tool

hf_tools = [load_tool(tool_name)
    for tool_name in ["document-question-answering",
    "image-captioning", "image-question-answering",
    'image-segmentation', "speech-to-text", "summarization",
    "text-classification", "text-question-answering",
    "translation", "huggingface-tools/text-to-image",
    "huggingface-tools/text-to-video", "text-to-speech",
    "huggingface-tools/text-download", "huggingface-tools/image-transformation"]]

from langchain_experimental.autonomous_agents import HuggingGPT
from langchain_openai import OpenAI

# %env OPENAI_API_BASE=http://localhost:8000/v1
llm = OpenAI(model_name="gpt-3.5-turbo")
agent = HuggingGPT(llm, hf_tools)
agent.run("please show me a video and an image of 'a boy is running'")
