# pip install langgraph langchain_openai
# export OPENAI_API_KEY=sk-...

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph

model = ChatOpenAI(temperature=0)
graph = MessageGraph()
graph.add_node("oracle", model)
graph.add_edge("oracle", END)
graph.set_entry_point("oracle")
runnable = graph.compile()
runnable.invoke(HumanMessage("What is 1 + 1?"))
# [HumanMessage(content='What is 1 + 1?'), AIMessage(content='1 + 1 equals 2.')]
