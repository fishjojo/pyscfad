from distutils.version import LooseVersion
import jax
import jax.ops
from jax import numpy as jnp
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)

# pylint: disable=no-member

index = jnp.index_exp
if LooseVersion(jax.__version__) < '0.2.22':
    index_update = jax.ops.index_update
    index_add = jax.ops.index_add
    index_mul = jax.ops.index_mul
else:
    def index_update(x, idx, y, indices_are_sorted=False, unique_indices=False):
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        return x.at[idx].set(y)

    def index_add(x, idx, y, indices_are_sorted=False, unique_indices=False):
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        return x.at[idx].add(y)

    def index_mul(x, idx, y, indices_are_sorted=False, unique_indices=False):
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        return x.at[idx].multiply(y)


del LooseVersion
