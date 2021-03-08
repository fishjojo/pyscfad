from jax import numpy as jnp

array = jnp.array

def asarray(*args, **kwargs):
    return jnp.asarray(*args, **kwargs)

def dot(*args, **kwargs):
    return jnp.dot(*args, **kwargs)

def einsum(*args, **kwargs):
    return jnp.einsum(*args, **kwargs)

def zeros(*args, **kwargs):
    return jnp.zeros(*args, **kwargs)
