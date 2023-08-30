from packaging.version import Version
import jax
import jax.ops
from jax import numpy as jnp
from pyscfad import config

_JaxArray = config.numpy_backend == 'jax'

# pylint: disable=no-member

if Version(jax.__version__) < Version('0.2.22'):
    _index_update = jax.ops.index_update
    _index_add = jax.ops.index_add
    _index_mul = jax.ops.index_mul
else:
    def _index_update(x, idx, y, indices_are_sorted=False, unique_indices=False):
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        return x.at[idx].set(y)

    def _index_add(x, idx, y, indices_are_sorted=False, unique_indices=False):
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        return x.at[idx].add(y)

    def _index_mul(x, idx, y, indices_are_sorted=False, unique_indices=False):
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        return x.at[idx].multiply(y)

class _Indexable(object):
    # pylint: disable=line-too-long
    """
    see https://github.com/google/jax/blob/97d00584f8b87dfe5c95e67892b54db993f34486/jax/_src/ops/scatter.py#L87
    """
    __slots__ = ()

    def __getitem__(self, idx):
        return idx

index = _Indexable()

def index_update(a, idx, value):
    if _JaxArray:
        a = _index_update(a, idx, value)
    else:
        a[idx] = value
    return a

def index_add(a, idx, value):
    if _JaxArray:
        a = _index_add(a, idx, value)
    else:
        a[idx] += value
    return a

def index_mul(a, idx, value):
    if _JaxArray:
        a = _index_mul(a, idx, value)
    else:
        a[idx] *= value
    return a
