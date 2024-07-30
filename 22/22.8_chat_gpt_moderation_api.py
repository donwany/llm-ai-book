from openai import OpenAI

client = OpenAI(api_key="sk-xxxxx")
response = client.moderations.create(input="I want to attack people in the mall and shoot them all.")
output = response.results[0]

# ----------------- OUTPUT ---------------------------
# {
#     "id": "modr-XXXXX",
#     "model": "text-moderation-007",
#     "results": [
#         {
#             "flagged": true,
#             "categories": {
#                 "sexual": false,
#                 "hate": false,
#                 "harassment": false,
#                 "self-harm": false,
#                 "sexual/minors": false,
#                 "hate/threatening": false,
#                 "violence/graphic": false,
#                 "self-harm/intent": false,
#                 "self-harm/instructions": false,
#                 "harassment/threatening": true,
#                 "violence": true
#             },
#             "category_scores": {
#                 "sexual": 1.2282071e-6,
#                 "hate": 0.010696256,
#                 "harassment": 0.29842457,
#                 "self-harm": 1.5236925e-8,
#                 "sexual/minors": 5.7246268e-8,
#                 "hate/threatening": 0.0060676364,
#                 "violence/graphic": 4.435014e-6,
#                 "self-harm/intent": 8.098441e-10,
#                 "self-harm/instructions": 2.8498655e-11,
#                 "harassment/threatening": 0.63055265,
#                 "violence": 0.99011886
#             }
#         }
#     ]
# }
