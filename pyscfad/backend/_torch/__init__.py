from types import ModuleType
try:
    import torch as torch
except ImportError as err:
    raise ImportError("Unable to import torch.") from err

from ..config import default_floatx
if default_floatx() == 'float64':
    torch.set_default_dtype(torch.float64)

from torch import (
    is_tensor as is_array,
)

from .._common import (
    class_as_pytree_node,
    custom_jvp,
)
from .numpy import (
    iscomplexobj,
)
from .core import (
    to_numpy,
    stop_gradient,
    vmap,
    jit,
)

class TorchBackend:
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

backend = TorchBackend(torch)
backend._cache['iscomplexobj'] = iscomplexobj

backend._cache['is_array'] = is_array
backend._cache['to_numpy'] = to_numpy
backend._cache['stop_gradient'] = stop_gradient
backend._cache['class_as_pytree_node'] = class_as_pytree_node
backend._cache['custom_jvp'] = custom_jvp
backend._cache['jit'] = jit
backend._cache['vmap'] = vmap

