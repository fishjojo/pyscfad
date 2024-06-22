import torch

def to_numpy(x):
    return x.numpy(force=True)

def stop_gradient(x):
    return x.detach()

def vmap(fun, in_axes=0, out_axes=0, chunk_size=None, signature=None):
    return torch.vmap(fun, in_dims=in_axes, out_dims=out_axes, chunk_size=chunk_size)

def jit(obj, **kwargs):
    # TODO make jit work
    #return torch.jit.script(obj, **kwargs)
    return obj
