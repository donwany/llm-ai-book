# # Clone the BMTools repository
# git clone git@github.com:OpenBMB/BMTools.git
#
# # Change directory to the cloned repository
# cd BMTools
#
# # Upgrade pip
# pip install --upgrade pip
#
# # Install the required dependencies
# pip install -r requirements.txt
#
# # Set up the BMTools package in development mode
# python setup.py develop


###############
# Single Tool
###############
from bmtools.agent.singletool import load_single_tools, STQuestionAnswerer

# Define the tool name and URL
tool_name, tool_url = 'klarna', 'https://www.klarna.com/'

# Load the single tool configuration
tool_name, tool_config = load_single_tools(tool_name, tool_url)
print(tool_name, tool_config)

# Initialize the single tool question answerer
stqa = STQuestionAnswerer()
agent = stqa.load_tools(tool_name, tool_config)

# Ask a question using the loaded tool
agent("{Your Question}")

###################
# Multiple Tools
###################
from bmtools.agent.tools_controller import load_valid_tools, MTQuestionAnswerer

# Define the tools mapping with their URLs
tools_mappings = {
    "klarna": "https://www.klarna.com/",
    "chemical-prop": "http://127.0.0.1:8079/tools/chemical-prop/",
    "wolframalpha": "http://127.0.0.1:8079/tools/wolframalpha/",
}

# Load the valid tools configurations
tools = load_valid_tools(tools_mappings)

# Initialize the multiple tools question answerer with OpenAI API key
qa = MTQuestionAnswerer(openai_api_key='your_openai_api_key', all_tools=tools)

# Build the agent runner
agent = qa.build_runner()

# Ask a question using the loaded tools
agent(
    "How many benzene rings are there in 9H-Carbazole-3-carboxaldehyde? "
    "and what is sin(x)*exp(x)'s plot, what is it integrated from 0 to 1?")

# Run the web demo
# python web_demo.py
