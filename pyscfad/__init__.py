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

"""
PySCF with auto-differentiation
"""
import sys
from pyscfad.version import __version__

from pyscfad._src._config import (
    config,
    config_update
)

# export backend.numpy to pyscfad namespace
# pylint: disable=useless-import-alias
from pyscfad.backend import (
    numpy as numpy,
    ops as ops,
    pytree as pytree,
)
sys.modules['pyscfad.numpy'] = numpy
sys.modules['pyscfad.ops'] = ops
sys.modules['pyscfad.pytree'] = pytree

del sys
