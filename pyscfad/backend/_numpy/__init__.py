from types import ModuleType
try:
    import numpy as np
except ImportError as err:
    raise ImportError("Unable to import numpy.") from err

from .._common import (
    stop_gradient,
    class_as_pytree_node,
    custom_jvp,
    jit, # TODO use numba
    index,
    index_update,
    index_add,
    index_mul,
)
from .core import (
    is_array,
    to_numpy,
    vmap,
)

class NumpyBackend:
    def __init__(self, package):
        self._pkg = package
        self._cache = {}

    def __getattr__(self, name):
        if name in self._cache:
            return self._cache[name]

        try:
            attr = getattr(self._pkg, name)
            if isinstance(attr, ModuleType):
                submodule = self.__class__(attr)
                self._cache[name] = submodule
                return submodule
            else:
                self._cache[name] = attr
                return attr
        except AttributeError as err:
            raise AttributeError(f"{self._pkg.__name__} has no attribute {name}") from err

backend = NumpyBackend(np)

# FIXME maybe separate ops from numpy
backend._cache['is_array'] = is_array
backend._cache['to_numpy'] = to_numpy
backend._cache['stop_gradient'] = stop_gradient
backend._cache['class_as_pytree_node'] = class_as_pytree_node
backend._cache['custom_jvp'] = custom_jvp
backend._cache['jit'] = jit
backend._cache['vmap'] = vmap
backend._cache['index'] = index
backend._cache['index_update'] = index_update
backend._cache['index_add'] = index_add
backend._cache['index_mul'] = index_mul

