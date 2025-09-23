# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import ModuleType
try:
    import jax
except ImportError as err:
    raise ImportError("Unable to import jax.") from err

from ..config import default_floatx
if default_floatx() == 'float64':
    jax.config.update("jax_enable_x64", True)

from .._common import (
    index,
)

from jax import (
    custom_jvp,
    jit,
    pure_callback,
    ShapeDtypeStruct,
)
from jax.lax import (
    stop_gradient,
    while_loop,
)
from .core import (
    is_array,
    to_numpy,
    vmap,
    index_update,
    index_add,
    index_mul,
)

from .pytree import (
    PytreeNode,
    class_as_pytree_node,
)

class JaxBackend:
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

backend = JaxBackend(jax.numpy)

backend._cache['is_array'] = is_array
backend._cache['to_numpy'] = to_numpy
backend._cache['stop_gradient'] = stop_gradient
backend._cache['custom_jvp'] = custom_jvp
backend._cache['jit'] = jit
backend._cache['pure_callback'] = pure_callback
backend._cache['ShapeDtypeStruct'] = ShapeDtypeStruct
backend._cache['vmap'] = vmap
backend._cache['while_loop'] = while_loop
backend._cache['index'] = index
backend._cache['index_update'] = index_update
backend._cache['index_add'] = index_add
backend._cache['index_mul'] = index_mul

backend._cache['class_as_pytree_node'] = class_as_pytree_node
backend._cache['PytreeNode'] = PytreeNode

