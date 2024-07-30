# Step 1: Create an Assistant
from openai import OpenAI

client = OpenAI()

# Step 1: Create an Assistant
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-turbo",
)

# Step 2: Create a Thread
thread = client.beta.threads.create()

# Step 3: Add a Message to the Thread
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
)

# Step 4: Create a Run
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions="Please address the user as Jane Doe. The user has a premium account."
)

if run.status == 'completed':
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    print(messages)
else:
    print(run.status)
