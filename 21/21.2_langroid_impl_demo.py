# Install necessary packages
# pip install langroid langroid[hf-embeddings]

# Export OpenAI API Key
# export OPENAI_API_KEY=your-key-here-without-quotes

import langroid as lr
import langroid.language_models as lm

# Set up LLM configuration
llm_cfg = lm.OpenAIGPTConfig(  # or OpenAIAssistant to use Assistant API
    chat_model=lm.OpenAIChatModel.GPT4_TURBO,  # or, e.g., "ollama/mistral"
)

# Use LLM directly
mdl = lm.OpenAIGPT(llm_cfg)
response = mdl.chat("What is the capital of Ontario?", max_tokens=10)
print(response)  # Print the response

# Use LLM in an Agent
agent_cfg = lr.ChatAgentConfig(llm=llm_cfg)
agent = lr.ChatAgent(agent_cfg)
print(agent.llm_response("What is the capital of Russia?"))
response = agent.llm_response("And Ghana?")  # Maintains conversation state
print(response)

# Wrap Agent in a Task to run an interactive loop with user (or other agents)
task = lr.Task(agent, name="Bot", system_message="You are a helpful assistant")
task.run("Hello")  # Kick off with user saying "Hello"

# 2-Agent chat loop: Teacher Agent asks questions to Student Agent
teacher_agent = lr.ChatAgent(agent_cfg)
teacher_task = lr.Task(
    teacher_agent, name="Teacher",
    system_message="""
    Ask your student concise numbered questions, and give feedback. Start with a question."""
)

student_agent = lr.ChatAgent(agent_cfg)
student_task = lr.Task(
    student_agent, name="Student",
    system_message="Concisely answer the teacher's questions.",
    single_round=True,
)

# Add the student task as a sub-task to the teacher task
teacher_task.add_sub_task(student_task)

# Run the teacher task, which will also run the student sub-task
teacher_task.run()
