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
