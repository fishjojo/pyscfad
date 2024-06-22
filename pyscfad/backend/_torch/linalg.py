import torch
from .core import convert_to_tensor

def cholesky(a, **kwargs):
    a = convert_to_tensor(a)
    return torch.linalg.cholesky(a, **kwargs)

def eigh(a, UPLO='L', **kwargs):
    a = convert_to_tensor(a)
    return torch.linalg.eigh(a, UPLO, **kwargs)

def inv(a, **kwargs):
    a = convert_to_tensor(a)
    return torch.linalg.inv(a, **kwargs)

def norm(x, ord=None, axis=None, keepdims=False, **kwargs):
    x = convert_to_tensor(x)
    return torch.linalg.norm(x, ord, axis, keepdims, **kwargs)
