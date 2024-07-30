import boto3
import json

# Create the AWS client for the Bedrock runtime with boto3
aws_client = boto3.client(service_name="bedrock-runtime")

# Input parameters for embed. In this example we are embedding hacker news post titles.
texts = ["Interesting (Non software) books?",
         "Non-tech books that have helped you grow professionally?",
         "I sold my company last month for $5m. What do I do with the money?",
         "How are you getting through (and back from) burning out?",
         "I made $24k over the last month. Now what?",
         "What kind of personal financial investment do you do?",
         "Should I quit the field of software development?"]
input_type = "clustering"
truncate = "NONE" # optional
model_id = "cohere.embed-english-v3" # or "cohere.embed-multilingual-v3"

# Create the JSON payload for the request
json_params = {
        'texts': texts,
        'truncate': truncate,
        "input_type": input_type
    }
json_body = json.dumps(json_params)
params = {'body': json_body, 'modelId': model_id,}

# Invoke the model and print the response
result = aws_client.invoke_model(**params)
response = json.loads(result['body'].read().decode())
print(response)
