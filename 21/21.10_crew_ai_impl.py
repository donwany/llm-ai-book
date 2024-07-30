# Install the required packages
# pip install crewai pandas
# pip install 'crewai[tools]'

import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

# Set the OpenAI API Key and Serper API Key environment variables
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
os.environ["SERPER_API_KEY"] = "Your_Key"  # serper.dev API key

# Initialize the search tool from SerperDevTool
search_tool = SerperDevTool()

# Define the Researcher agent
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI and data science',
    backstory="""You work at a leading tech think tank.
    Your expertise lies in identifying emerging trends.
    You have a knack for dissecting complex data and presenting actionable insights.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool]
)

# Define the Writer agent
writer = Agent(
    role='Tech Content Strategist',
    goal='Craft compelling content on tech advancements',
    backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
    You transform complex concepts into compelling narratives.""",
    verbose=True,
    allow_delegation=True
)

# Create a task for the Researcher agent
task1 = Task(
    description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024. Identify key trends, breakthrough technologies, and potential industry impacts.""",
    expected_output="Full analysis report in bullet points",
    agent=researcher
)

# Create a task for the Writer agent
task2 = Task(
    description="""Using the insights provided, develop an engaging blog post that highlights the most significant AI advancements.
    Your post should be informative yet accessible, catering to a tech-savvy audience. Make it sound cool, avoid complex words so it doesn't sound like AI.""",
    expected_output="Full blog post of at least 4 paragraphs",
    agent=writer
)

# Instantiate the crew with a sequential process
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2  # You can set it to 1 or 2 for different logging levels
)

# Kickoff the crew to start working on the tasks
result = crew.kickoff()

# Print the result
print("######################")
print(result)
