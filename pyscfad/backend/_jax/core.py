import jax
from jax import numpy as jnp

def is_array(x):
    return isinstance(x, jax.Array)

def to_numpy(x):
    x = jax.lax.stop_gradient(x)
    return x.__array__()

def vmap(fun, in_axes=0, out_axes=0, chunk_size=None, signature=None):
    return jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)

# TODO deprecate these
def index_update(x, idx, y):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return x.at[idx].set(y)

def index_add(x, idx, y):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return x.at[idx].add(y)

def index_mul(x, idx, y):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return x.at[idx].multiply(y)

