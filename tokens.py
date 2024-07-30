from transformers import AutoModelForCausalLM, AutoTokenizer
prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."
# Placeholder definition. The next code blocks show the actual generation


def generate(prompt, number_of_tokens):
    # TODO: pass prompt to language model, and return the text it generates
    pass

output = generate(prompt, 10)
print(output)

# openchat is a 13B LLM
model_name = "openchat/openchat"
# If your environment does not have the required resources to run this model
# then try a smaller model like "gpt2" or "openlm-research/open_llama_3b"
# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load a language model
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."
# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
# Generate the text
generation_output = model.generate(input_ids=input_ids, max_new_tokens=256)
# Print the output
print(tokenizer.decode(generation_output[0]))

for id in input_ids[0]:
    print(tokenizer.decode(id))
