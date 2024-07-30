from crewai_tools import ScrapeWebsiteTool

# Initialize the ScrapeWebsiteTool with the website URL
tool = ScrapeWebsiteTool(website_url='https://www.example.com')

# Extract the text from the site
text = tool.run()

# Print the extracted text
print("Extracted Text from Website:")
print(text)
