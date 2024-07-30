# Install the necessary library
# pip install guidance

from guidance import models, gen


# ------------------ LlamaCpp Model Example ------------------
def llama_cpp_example(path):
    # Load the LlamaCpp model
    llama2 = models.LlamaCpp(path)

    # Generate a response with the model
    response = llama2 + 'Do you want a joke or a poem? ' + gen(stop='.')
    print("LlamaCpp Response:", response)


# ------------------ OpenAI Model Examples ------------------
def openai_examples():
    # Load the OpenAI Curie model
    curie = models.OpenAI("text-curie-001")

    # Generate a response with the Curie model
    curie_response = curie + "The smallest cats are" + gen(stop=".")
    print("Curie Response:", curie_response)

    # Load the OpenAI GPT-3.5-turbo model for chat
    gpt = models.OpenAI("gpt-3.5-turbo")

    # Generate a response with chat models
    with models.system():
        lm = gpt + "You are a cat expert."
    with models.user():
        lm += "What are the smallest cats?"
    with models.assistant():
        lm += gen("answer", stop=".")
    chat_response = lm
    print("GPT-3.5-turbo Chat Response:", chat_response)


# ------------------ VertexAI Model Example ------------------
def vertex_ai_example():
    # Load the VertexAI model
    palm2 = models.VertexAI("text-bison@001")

    # Generate a response with VertexAI
    with models.instruction():
        lm = palm2 + "What is one funny fact about Seattle?"
    vertex_response = lm + gen("fact", max_tokens=100)
    print("VertexAI Response:", vertex_response)


# Execute examples
llama_cpp_example("path_to_llama_cpp_model")
openai_examples()
vertex_ai_example()
