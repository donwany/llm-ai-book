import torch


# absmax
def absmax_quantize(X):
    # Calculate scale
    scale = 127 / torch.max(torch.abs(X))
    # Quantize
    X_quantized = (scale * X).round()
    # Dequantize
    X_dequantized = X_quantized / scale
    # Convert to int8 tensor
    X_quantized_int8 = X_quantized.to(torch.int8)
    return X_quantized_int8, X_dequantized


# zero-point
def zeropoint_quantize(X):
    # Calculate value range (denominator)
    x_range = torch.max(X) - torch.min(X)
    x_range = 1 if x_range == 0 else x_range
    # Calculate scale
    scale = 255 / x_range
    # Shift by zero-point
    zeropoint = (-scale * torch.min(X) - 128).round()
    # Scale and round the inputs
    X_quantized = torch.clip((X * scale + zeropoint).round(), -128, 127)
    # Dequantize
    X_dequantized = (X_quantized - zeropoint) / scale
    # Convert to int8 tensor
    X_quantized_int8 = X_quantized.to(torch.int8)
    return X_quantized_int8, X_dequantized