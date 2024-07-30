import torch


def softmax(x, dim=-1):
    # Apply the softmax function along the specified dimension
    # Subtract the maximum value for numerical stability
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    # Normalize by the sum along the specified dimension
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


# Generate random data
torch.manual_seed(0)  # For reproducibility
data = torch.randn(5, 3)  # 5 samples, 3 classes

# Apply the softmax function
softmax_output = softmax(x=data)

print("Random Data:")
print(data)
print("\nSoftmax Output:")
print(softmax_output)
print("\nSum of probabilities for each sample (should be close to 1):")
print(torch.sum(softmax_output, dim=1))
