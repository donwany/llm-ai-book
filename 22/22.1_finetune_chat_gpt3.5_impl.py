# pip install --upgrade openai
from openai import OpenAI
import time


def create_finetuning_file(client, filename):
    # Create file for fine-tuning
    file = client.files.create(
        file=open(filename, "rb"), purpose="fine-tune")
    return file


def create_finetuning_job(client, training_file_id, model_name):
    # Create a fine-tuning job
    job = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=model_name
    )
    return job


def check_job_status(client, job_id):
    # Check the status of the fine-tuning job
    job = client.fine_tuning.jobs.retrieve(job_id)
    return job


def main():
    client = OpenAI(api_key="sk-xxxxx")
    # Create file for fine-tuning
    file = create_finetuning_file(client, "fine-tuning-dataset.jsonl")
    print(f"File ID: {file.id}, File Status: {file.status}")

    # Create a fine-tuning job
    job = create_finetuning_job(client, file.id, "gpt-3.5-turbo-1106")
    print(f"Job ID: {job.id}, Job Status: {job.status}")

    while True:
        time.sleep(10)
        job = check_job_status(client, job.id)
        if job.status == 'succeeded':
            model_name = f"ft-{job.model}:suffix:{job.id}"
            print(f"Job ID: {job.id}, Final Job Status: {job.status}")
            print(f"Model Name: {model_name}")
            break
        elif job.status == 'failed':
            print(f"Job ID: {job.id}, Final Job Status: {job.status}")
            break
        else:
            print(f"Checking... Job ID: {job.id}, Current Job Status: {job.status}")


if __name__ == "__main__":
    main()

# ------------- INFERENCE ------------------------------------
from openai import OpenAI

client = OpenAI(api_key="sk-xxxxx")
response = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-1106:<USERNAME>::8QUpHZjq",
    messages=[
        {"role": "system", "content": "Elon Musk is a factual chatbot with a sense of humor."},
        {"role": "user", "content": "How far is the sun?"},
    ]
)
print(response.choices[0].message.content)
