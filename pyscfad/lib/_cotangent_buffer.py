# Copyright 2026 The PySCFAD Authors
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

'''
In-place accumulation of the cotangent of a large array, e.g., the
density fitting tensor.

The cotangent of an array used by several custom VJPs is the sum of their
contributions, and JAX allocates every contribution and every partial sum,
each as large as the array. Instead, the custom VJP producing the array
registers it in its forward rule (:func:`register`). The backward rule of a
consumer then adds its contribution to the buffer of the array
(:func:`get`) and returns ``None`` (zero) as its cotangent. The backward
rule of the producer, which runs after those of all its consumers, takes the
buffer (:func:`pop`) along with the cotangent passed by JAX.

A consumer must only use the buffer for an array that it differentiates
(perturbed) and whose cotangent goes unchanged to the producer, as in a
reverse pass through the producer, including the nested VJPs whose
cotangents are returned to it, e.g., the implicit differentiation of SCF.
'''

import weakref
import numpy

_buffers = {}

def _owner(x):
    # the array owning the data of x, kept alive by all the views of x
    while isinstance(x.base, numpy.ndarray):
        x = x.base
    return x

class _Buffer:
    def __init__(self, x):
        self.ref = weakref.ref(_owner(x))
        self.shape = x.shape
        self.dtype = x.dtype
        self.data = None

def _data_ptr(x):
    return x.__array_interface__['data'][0]

def register(x):
    '''Register the numpy array ``x`` for the in-place accumulation of
    its cotangent, until ``x`` is deleted.'''
    key = _data_ptr(x)
    _buffers[key] = _Buffer(x)
    weakref.finalize(_owner(x), _buffers.pop, key, None)

def _lookup(x):
    if not _buffers or not isinstance(x, numpy.ndarray):
        return None
    buf = _buffers.get(_data_ptr(x))
    if (buf is None or buf.ref() is None
            or buf.shape != x.shape or buf.dtype != x.dtype):
        return None
    return buf

def registered(x):
    '''Whether the cotangent of ``x`` is accumulated in a buffer'''
    return _lookup(x) is not None

def get(x):
    '''The buffer to add the cotangent of ``x`` to, or ``None`` if ``x``
    is not registered.'''
    buf = _lookup(x)
    if buf is None:
        return None
    if buf.data is None:
        buf.data = numpy.zeros(buf.shape, dtype=buf.dtype)
    return buf.data

def pop(x):
    '''The cotangent of ``x`` accumulated so far, or ``None`` if nothing
    was added, and empty the buffer for the next reverse pass.'''
    buf = _lookup(x)
    if buf is None:
        return None
    data, buf.data = buf.data, None
    return data
