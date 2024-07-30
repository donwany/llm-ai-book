# pip install guardrails-ai

# Install regex match guardrail from Guardrails Hub
# guardrails hub install hub://guardrails/regex_match

# Import and Use Guard and Validator
from guardrails.hub import RegexMatch
from guardrails import Guard

# Initialize the Guard with RegexMatch
guard = Guard().use(RegexMatch(regex="^[A-Z][a-z]*$"))

# Test guardrail validation
guard.parse("Caesar")  # Guardrail Passes
guard.parse("Caesar is a great leader")  # Guardrail Fails

# -------- Run Multiple Guardrails -----------
from guardrails.hub import CompetitorCheck, ToxicLanguage

# Initialize Guard with multiple validators
competitors = ["Apple", "Samsung"]
guard = Guard().use(
    CompetitorCheck(competitors=competitors)
).use(
    ToxicLanguage(validation_method='sentence', threshold=0.5)
)

# Test guardrail validation
guard.validate("My favorite phone is BlackBerry.")  # Guardrail Passes

# ------------ Create a Guard from Installed Guardrail -----------
import openai
from guardrails import Guard
from guardrails.hub import RegexMatch
from pydantic import BaseModel


# Define a Pydantic model for the expected output
class Pet(BaseModel):
    pet_type: str
    name: str


# Define the prompt and create a Guard instance
prompt = """
    What kind of pet should I get and what should I name it?
    ${gr.complete_json_suffix_v2}
"""
guard = Guard.from_pydantic(output_class=Pet, prompt=prompt)

# Use the Guard instance with OpenAI API
validated_output, *rest = guard(
    llm_api=openai.Completion.create,
    engine="gpt-3.5-turbo-instruct"
)

# Print the validated output
print(validated_output)

# { "pet_type": "dog", "name": "Buddy"}
