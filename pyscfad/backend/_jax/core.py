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

import jax
from jax import numpy as jnp

def is_array(x):
    return isinstance(x, jax.Array)

def to_numpy(x):
    if is_array(x):
        x = jax.lax.stop_gradient(x)
        if is_array(x):
            x = x.__array__()
    return x

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

