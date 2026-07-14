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

import numpy
from pyscfad import numpy as np
from pyscfad.backend.config import default_floatx


def test_floatx_dtype():
    assert np.floatx == numpy.dtype(default_floatx())
    assert np.floatx in (numpy.dtype('float32'), numpy.dtype('float64'))


def test_create_array_with_floatx():
    a = np.zeros((2, 2), dtype=np.floatx)
    assert a.dtype == np.floatx
    b = numpy.zeros(2, dtype=np.floatx)
    assert b.dtype == np.floatx


def test_astype_floatx():
    other = numpy.float32 if np.floatx == numpy.float64 else numpy.float64
    a = np.zeros(2, dtype=other).astype(np.floatx)
    assert a.dtype == np.floatx


def test_complexx_dtype():
    if np.floatx == numpy.dtype('float32'):
        assert np.complexx == numpy.dtype('complex64')
    else:
        assert np.complexx == numpy.dtype('complex128')
    # complexx is the complex counterpart of floatx
    assert numpy.dtype(numpy.result_type(np.floatx, numpy.complex64)) == np.complexx
    a = np.zeros(2, dtype=np.floatx).astype(np.complexx)
    assert a.dtype == np.complexx
    assert a.real.dtype == np.floatx
