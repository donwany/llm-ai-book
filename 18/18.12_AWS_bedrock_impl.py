import json
import boto3
import botocore

boto3_bedrock = boto3.client('bedrock-runtime')
prompt_data = """Command: Write me a blog about making strong business decisions as a leader.
Blog:
"""

try:
    # Prepare the request body
    body = json.dumps({
        "inputText": prompt_data,
        "textGenerationConfig": {
            "topP": 0.95,
            "temperature": 0.2
        }
    })

    # Set up the model ID and other parameters
    model_id = "amazon.titan-tg1-large"
    accept = "application/json"
    content_type = "application/json"

    # Invoke the model
    response = boto3_bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept=accept,
        contentType=content_type
    )

    # Parse the response
    response_body = json.loads(response.get("body").read())
    print(response_body.get("results")[0].get("outputText"))

except botocore.exceptions.ClientError as error:
    # Handle access denied error specifically
    if error.response['Error']['Code'] == 'AccessDeniedException':
        print(f"\x1b[41m{error.response['Error']['Message']}"
              "\nTo troubleshoot this issue please refer to the following resources."
              "\nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html"
              "\nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
    else:
        raise error
