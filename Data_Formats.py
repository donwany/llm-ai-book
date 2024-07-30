# import numpy as np
# from scipy.special import softmax
#
# # Step 1: Input: 3 inputs, d_model=4
# x = np.array([[1.0, 0.0, 1.0, 0.0],
#               [0.0, 2.0, 0.0, 2.0],
#               [1.0, 1.0, 1.0, 1.0]])
#
# # Step 2: weights 3 dimensions x d_model=4
# w_query = np.array([1, 0, 1],
#                    [1, 0, 0],
#                    [0, 0, 1],
#                    [0, 1, 1]])
# w_key = np.array([[0, 0, 1],
#                   [1, 1, 0],
#                   [0, 1, 0],
#                   [1, 1, 0]])
# w_value = np.array([[0, 2, 0],
#                     [0, 3, 0],
#                     [1, 0, 3],
#                     [1, 1, 0]])
#
# # Step 3: Matrix Multiplication to obtain Q,K,V
# ## Query: x * w_query
# Q = np.matmul(x, w_query)
# ## Key: x * w_key
# K = np.matmul(x, w_key)
# ## Value: x * w_value
# V = np.matmul(x, w_value)
#
# # Step 4: Scaled Attention Scores
# ## Square root of the dimensions
# k_d = 1
# attention_scores = (Q @ K.transpose()) / k_d
#
# # Step 5: Scaled softmax attention scores for each vector
# attention_scores[0] = softmax(attention_scores[0])
# attention_scores[1] = softmax(attention_scores[1])
# attention_scores[2] = softmax(attention_scores[2])
#
# # Step 6: attention value obtained by score1/k_d * V
# attention1 = attention_scores[0].reshape(-1, 1)
# attention1 = attention_scores[0][0] * V[0]
# attention2 = attention_scores[0][1] * V[1]
# attention3 = attention_scores[0][2] * V[2]
#
# # Step 7: summed the results to create the first line of the output matrix
# attention_input1 = attention1 + attention2 + attention3
#
# # Step 8: Step 1 to 7 for inputs 1 to 3
# ## Because this is just a demo, we’ll do a random matrix of the right dimensions
# attention_head1 = np.random.random((3, 64))
#
# # Step 9: We train all 8 heads of the attention sub-layer using steps 1 through 7
# ## Again, it’s a demo
# z0h1 = np.random.random((3, 64))
# z1h2 = np.random.random((3, 64))
# z2h3 = np.random.random((3, 64))
# z3h4 = np.random.random((3, 64))
# z4h5 = np.random.random((3, 64))
# z5h6 = np.random.random((3, 64))
# z6h7 = np.random.random((3, 64))
# z7h8 = np.random.random((3, 64))
#
# # Step 10: Concatenate heads 1 through 8 to get the original 8x64 output dimension of the model
# Output_attention = np.hstack((z0h1, z1h2, z2h3, z3h4, z4h5, z5h6, z6h7, z7h8))
#
# # Here’s a function that performs all of these steps:
#
#
# def dot_product_attention(query, key, value, mask, scale=True):
#     assert query.shape[-1] == key.shape[-1] == value.shape[-1], “q, k, v have different dimensions!”
#     if scale:
#         depth = query.shape[-1]
#     else:
#         depth = 1
#     dots = np.matmul(query, np.swapaxes(key, -1, -2)) / np.sqrt(depth)
#     if mask is not None:
#         dots = np.where(mask, dots, np.full_like(dots, -1e9))
#     logsumexp = scipy.special.logsumexp(dots, axis=-1, keepdims=True)
#     dots = np.exp(dots - logsumexp)
#     attention = np.matmul(dots, value)
#     return attention
#
#
# # Here’s a function that performs the previous steps but adds causality in masking
# def masked_dot_product_self_attention(q, k, v, scale=True):
#     mask_size = q.shape[-2]
#     mask = np.tril(np.ones((1, mask_size, mask_size), dtype=np.bool_), k=0)
#     return DotProductAttention(q, k, v, mask, scale=scale)