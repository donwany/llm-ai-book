import os
import openai
import json
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key from the environment variables
openai.api_key = os.environ['OPENAI_API_KEY']

def get_completion(messages, model="gpt-3.5-turbo-1106", temperature=0, max_tokens=300, tools=None, tool_choice=None):
    """
    Get a completion response from the OpenAI API.
    Args:
        messages (list): List of message dictionaries for the chat.
        model (str): The model to use for the chat.
        temperature (float): Sampling temperature to use.
        max_tokens (int): The maximum number of tokens to generate.
        tools (list): Optional list of tools/functions to be used by the model.
        tool_choice (str): Optional specific tool to use.

    Returns:
        dict: The response message from the API.
    """
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice
    )
    return response.choices[0].message

def get_current_weather(location, unit="fahrenheit"):
    """
    Get the current weather in a given location.
    Args:
        location (str): The location for which to get the weather.
        unit (str): The unit of temperature (celsius or fahrenheit).
    Returns:
        str: JSON string containing the weather information.
    """
    weather = {"location": location, "temperature": "50", "unit": unit}
    return json.dumps(weather)

# Define the function to be used as a tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. Dallas, TX"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Message input from the user
messages = [
    {
        "role": "user",
        "content": "What is the weather like in London?"
    }
]

# Get the completion from the model, using the defined tools
response = get_completion(messages, tools=tools)
print(response)

# Parse the arguments for the function call from the response
args = json.loads(response.tool_calls[0].function.arguments)

# Call the function with the parsed arguments
get_current_weather(**args)
