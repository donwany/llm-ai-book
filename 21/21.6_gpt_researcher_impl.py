# # Export your OpenAI API key
# export OPENAI_API_KEY={Your OpenAI API Key here}
#
# # Export your Tavily API key
# export TAVILY_API_KEY={Your Tavily API Key here}
#
# # Clone the gpt-researcher repository
# git clone https://github.com/assafelovic/gpt-researcher.git
#
# # Change directory to the cloned repository
# cd gpt-researcher
#
# # Install required packages
# python -m pip install -r requirements.txt
#
# # Start the server
# python -m uvicorn main:app --reload
#
# # Visit http://localhost:8000 in any web browser and explore your research!

# Import necessary modules from gpt_researcher
from gpt_researcher import GPTResearcher
import asyncio


# Define the main asynchronous function
async def main():
    """
    This is a sample script that shows how to run a research report.
    """
    # Define the query
    query = "What happened in the latest burning man floods?"

    # Define the report type
    report_type = "research_report"

    # Initialize the GPTResearcher with the given query and report type
    researcher = GPTResearcher(query=query, report_type=report_type, config_path=None)

    # Conduct research on the given query
    researcher.conduct_research()

    # Write the report and await the result
    report = await researcher.write_report()

    # Return the generated report
    return report


# Execute the main function
if __name__ == "__main__":
    asyncio.run(main())
