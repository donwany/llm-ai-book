# Install the necessary library
# pip install --upgrade cohere-aws
# Note: Restart the kernel after upgrading the package

import boto3
from cohere_aws import Client

# Define the model package identifier
cohere_package = "cohere-command-v16-0bf50631cbfc34c8aaef615865a57cc2"

# Mapping of AWS regions to their respective model package ARNs
model_package_map = {
    "us-east-1": f"arn:aws:sagemaker:us-east-1:865070037744:model-package/{cohere_package}",
    "us-east-2": f"arn:aws:sagemaker:us-east-2:057799348421:model-package/{cohere_package}",
    "us-west-1": f"arn:aws:sagemaker:us-west-1:382657785993:model-package/{cohere_package}",
    "us-west-2": f"arn:aws:sagemaker:us-west-2:594846645681:model-package/{cohere_package}",
    "ca-central-1": f"arn:aws:sagemaker:ca-central-1:470592106596:model-package/{cohere_package}",
    "eu-central-1": f"arn:aws:sagemaker:eu-central-1:446921602837:model-package/{cohere_package}",
    "eu-west-1": f"arn:aws:sagemaker:eu-west-1:985815980388:model-package/{cohere_package}",
    "eu-west-2": f"arn:aws:sagemaker:eu-west-2:856760150666:model-package/{cohere_package}",
    "eu-west-3": f"arn:aws:sagemaker:eu-west-3:843114510376:model-package/{cohere_package}",
    "eu-north-1": f"arn:aws:sagemaker:eu-north-1:136758871317:model-package/{cohere_package}",
    "ap-southeast-1": f"arn:aws:sagemaker:ap-southeast-1:192199979996:model-package/{cohere_package}",
    "ap-southeast-2": f"arn:aws:sagemaker:ap-southeast-2:666831318237:model-package/{cohere_package}",
    "ap-northeast-2": f"arn:aws:sagemaker:ap-northeast-2:745090734665:model-package/{cohere_package}",
    "ap-northeast-1": f"arn:aws:sagemaker:ap-northeast-1:977537786026:model-package/{cohere_package}",
    "ap-south-1": f"arn:aws:sagemaker:ap-south-1:077584701553:model-package/{cohere_package}",
    "sa-east-1": f"arn:aws:sagemaker:sa-east-1:270155090741:model-package/{cohere_package}",
}

# Get the current AWS region
region = boto3.Session().region_name

# Check if the current region is supported
if region not in model_package_map:
    raise Exception(f"Current boto3 session region {region} is not supported.")

# Retrieve the model package ARN for the current region
model_package_arn = model_package_map[region]

# Initialize the Cohere client for the current region
co = Client(region_name=region)

# Create an endpoint for the Cohere model
co.create_endpoint(
    arn=model_package_arn,
    endpoint_name="cohere-command",
    instance_type="ml.p4d.24xlarge",
    n_instances=1
)


# Define a function to generate text using the Cohere model
def generate_text(prompt, max_tokens=100, temperature=0.9, stream=False):
    """
    Generate text using the Cohere model.

    Args:
        prompt (str): The prompt for text generation.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The sampling temperature.
        stream (bool): Whether to stream the response.

    Returns:
        str: The generated text.
    """
    response = co.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream
    )
    return response.generations[0]["text"]


# Generate and print text for various prompts
prompts = {
    "LinkedIn Post": "Write a LinkedIn post about starting a career in tech:",
    "Product Description": "Write a creative product description for a wireless headphone product named the CO-1T, with the keywords bluetooth, wireless, fast charging for a software developer who works in noisy offices, and describe benefits of this product.",
    "Blog Post Paragraph": "Write a body paragraph about Shopify as a great case study in a blog post titled 'Tips from the most successful companies'.",
    "Cold Outreach Email": "Write a cold outreach email introducing myself as Susan, a business development manager at CoolCompany, to Amy who is a product manager at Microsoft asking if they'd be interested in speaking about an integration to add autocomplete to Microsoft Office."
}

for title, prompt in prompts.items():
    print(f"\n{title}:\n{generate_text(prompt)}")

# Delete the endpoint
co.delete_endpoint()

# Close the Cohere client
co.close()
