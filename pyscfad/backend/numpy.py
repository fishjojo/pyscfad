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
import numpy

def __getattr__(name):
    return getattr(get_backend(), name)

def safe_sqrt(x, thresh=0.0, fill_value=0.0):
    """Element-wise square root of the input
    with zero derivative at zero.

    Parameters:
        x: input array or scalar.
        thresh: elements with absolute values smaller than ``thresh``
            are treated as zeros.
        fill_value: fake output value at ``x = 0``.

    Returns:
        An array containing the element-wise square root of ``x``.

    Notes:
        ``grad(sqrt)(0) = inf``, while ``grad(safe_sqrt)(0) = 0``.
    """
    sqrt = getattr(get_backend(), "sqrt")
    where = getattr(get_backend(), "where")
    cond = abs(x) <= thresh
    x_safe = where(cond, 1., x)
    out = where(cond, fill_value, sqrt(x_safe))
    return out

def safe_reciprocal(x, thresh=0.0, fill_value=numpy.inf):
    """Element-wise reciprocal of the input
    with zero derivative at zeros.

    Parameters:
        x: input array or scalar.
        thresh: elements with absolute values smaller than ``thresh``
            are treated as zeros.
        fill_value: fake output value at ``x = 0``.

    Returns:
        An array containing the element-wise reciprocal of ``x``.
    """
    reciprocal = getattr(get_backend(), "reciprocal")
    where = getattr(get_backend(), "where")
    cond = abs(x) <= thresh
    x_safe = where(cond, 1., x)
    out = where(cond, fill_value, reciprocal(x_safe))
    return out
