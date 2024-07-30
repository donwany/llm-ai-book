import boto3
import json

# Create the AWS client for the Bedrock runtime with boto3
aws_client = boto3.client(service_name="bedrock-runtime")

# Create the JSON payload for the request
json_params = {'prompt': "Write a LinkedIn post about starting a career in tech:"}
params = {'body': json.dumps(json_params), 'modelId': 'cohere.command-text-v14', }

# Invoke the model and print the response
result = aws_client.invoke_model(**params)
response = json.loads(result['body'].read().decode())
print(response)
