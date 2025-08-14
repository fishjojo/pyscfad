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

from pyscfad import numpy as np
from pyscfad.soscf.ciah import extract_rotation

def rotate_mo1(mo_coeff, x):
    u = extract_rotation(x)
    return np.dot(mo_coeff, u)

def rotate_mo1_ov(mo_coeff, x, nocc):
    u = extract_rotation(x)
    u = u.at[:nocc,:nocc].set(0)
    u = u.at[nocc:,nocc:].set(0)
    return np.dot(mo_coeff, u)
