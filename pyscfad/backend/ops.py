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

from .config import get_backend

__all__ = [
    'is_array',
    'isarray',
    'is_tensor',
    'to_numpy',
    'stop_gradient',
    'stop_grad',
    'stop_trace',
    'custom_jvp',
    'jit',
    'vmap',
    'while_loop',
    'index',
    'index_update',
    'index_add',
    'index_mul',
]

def __getattr__(name):
    return getattr(get_backend(), name)

def is_array(x):
    return get_backend().is_array(x)

is_tensor = isarray = is_array

def to_numpy(x):
    return get_backend().to_numpy(x)

def stop_gradient(x):
    return get_backend().stop_gradient(x)

stop_grad = stop_gradient

def stop_trace(fn):
    """Convenient wrapper to call functions with arguments
    detached from the graph.
    """
    def wrapped_fn(*args, **kwargs):
        args_no_grad = [stop_grad(arg) for arg in args]
        kwargs_no_grad = {k : stop_grad(v) for k, v in kwargs.items()}
        return fn(*args_no_grad, **kwargs_no_grad)
    return wrapped_fn

def jit(obj, **kwargs):
    return get_backend().jit(obj, **kwargs)

def vmap(fun, in_axes=0, out_axes=0, chunk_size=None, signature=None):
    return get_backend().vmap(fun, in_axes=in_axes, out_axes=out_axes,
                              chunk_size=chunk_size, signature=signature)

def index_update(x, idx, y):
    return get_backend().index_update(x, idx, y)

def index_add(x, idx, y):
    return get_backend().index_add(x, idx, y)

def index_mul(x, idx, y):
    return get_backend().index_mul(x, idx, y)

