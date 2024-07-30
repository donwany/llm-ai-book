# pip install ctranslate2
import transformers
import ctranslate2

# Load the translator with a pre-trained translation model
translator = ctranslate2.Translator("ende_ctranslate2/", device="cpu")

# Translate a batch of tokens
results = translator.translate_batch([["_H", "ello", "_world", "!"]])

# Print the translated result
print(results[0].hypotheses[0])

# Load the text generation model
generator = ctranslate2.Generator("stablelm-ct2/", device="cpu")

# Load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")

# Define system prompt and encode it
system_prompt = """# StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""
system_prompt_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(system_prompt))

# Define the user prompt and encode it
prompt = "<|USER|>What’s your mood today?<|ASSISTANT|>"
prompt_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))

# Generate text based on the prompt and system prompt
step_results = generator.generate_tokens(
    prompt=prompt_tokens,
    static_prompt=system_prompt_tokens,
    max_length=512,
    sampling_topk=10,
    sampling_temperature=0.7,
    end_token=[50278, 50279, 50277, 1, 0],
)
# Repo: https://github.com/OpenNMT/CTranslate2.git
