from openai import OpenAI

client = OpenAI(api_key="sk-xxxxx")
response = client.embeddings.create(
    input="Your text string goes here",
    model="text-embedding-3-small"
)
print(response.data[0].embedding)

# ---------------- OUTPUT -----------------------------
# {
#     "object": "list",
#     "data": [{
#         "object": "embedding",
#         "index": 0,
#         "embedding": [
#             -0.006929283495992422,
#             -0.005336422007530928,
#             ... (omitted for spacing)
#             -4.547132266452536e-05,
#             -0.024047505110502243
#         ]
#     }],
#     "model": "text-embedding-3-small",
#     "usage": {
#         "prompt_tokens": 5,
#         "total_tokens": 5
#     }
# }
