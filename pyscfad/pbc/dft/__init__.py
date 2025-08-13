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

from pyscfad.pbc.dft import rks
from pyscfad.pbc.dft import krks

def RKS(cell, *args, **kwargs):
    return rks.RKS(cell, *args, **kwargs)

def KRKS(cell, *args, **kwargs):
    return krks.KRKS(cell, *args, **kwargs)
