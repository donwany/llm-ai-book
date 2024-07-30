# pip install -U pymemgpt
# git clone git@github.com:cpacker/MemGPT.git
# cd MemGPT
# memgpt run

from memgpt import create_client

# Connect to the server as a user
client = create_client()

# Create an agent
agent_info = client.create_agent(
    name="my_agent",
    persona="You are a friendly agent.",
    human="Bob is a friendly human."
)

# Send a message to the agent
messages = client.user_message(agent_id=agent_info.id, message="Hello, agent!")
