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

import numpy
from pyscf.data.nist import BOHR

__all__ = ["ATOMIC", "COV_D3"]

AA2AU = 1 / BOHR

ATOMIC = AA2AU * numpy.array([
    0.00,  # dummy
    0.32,0.37,  # H,He
    1.30,0.99,0.84,0.75,0.71,0.64,0.60,0.62,  # Li-Ne
    1.60,1.40,1.24,1.14,1.09,1.04,1.00,1.01,  # Na-Ar
    2.00,1.74,  # K,Ca
    1.59,1.48,1.44,1.30,1.29,  # Sc-
    1.24,1.18,1.17,1.22,1.20,  # -Zn
    1.23,1.20,1.20,1.18,1.17,1.16,  # Ga-Kr
    2.15,1.90,  # Rb,Sr
    1.76,1.64,1.56,1.46,1.38,  # Y-
    1.36,1.34,1.30,1.36,1.40,  # -Cd
    1.42,1.40,1.40,1.37,1.36,1.36,  # In-Xe
    2.38,2.06,  # Cs,Ba
    1.94,1.84,1.90,1.88,1.86,1.85,1.83,  # La-Eu
    1.82,1.81,1.80,1.79,1.77,1.77,1.78,  # Gd-Yb
    1.74,1.64,1.58,1.50,1.41,  # Lu-
    1.36,1.32,1.30,1.30,1.32,  # -Hg
    1.44,1.45,1.50,1.42,1.48,1.46,  # Tl-Rn
    2.42,2.11,  # Fr,Ra
    2.01,1.90,1.84,1.83,1.80,1.80,1.73,  # Ac-Am
    1.68,1.68,1.68,1.65,1.67,1.73,1.76,  # Cm-No
    1.61,1.57,1.49,1.43,1.41,  # Lr-
    1.34,1.29,1.28,1.21,1.22,   # -Cn
    1.36,1.43,1.62,1.75,1.65,1.57,  # Nh-Og
])
"""Atomic radii."""

COV_2009 = AA2AU * numpy.array([
    0.00,  # None
    0.32,0.46,  # H,He
    1.20,0.94,0.77,0.75,0.71,0.63,0.64,0.67,  # Li-Ne
    1.40,1.25,1.13,1.04,1.10,1.02,0.99,0.96,  # Na-Ar
    1.76,1.54,  # K,Ca
    1.33,1.22,1.21,1.10,1.07,  # Sc-
    1.04,1.00,0.99,1.01,1.09,  # -Zn
    1.12,1.09,1.15,1.10,1.14,1.17,  # Ga-Kr
    1.89,1.67,  # Rb,Sr
    1.47,1.39,1.32,1.24,1.15,  # Y-
    1.13,1.13,1.08,1.15,1.23,  # -Cd
    1.28,1.26,1.26,1.23,1.32,1.31,  # In-Xe
    2.09,1.76,  # Cs,Ba
    1.62,1.47,1.58,1.57,1.56,1.55,1.51,  # La-Eu
    1.52,1.51,1.50,1.49,1.49,1.48,1.53,  # Gd-Yb
    1.46,1.37,1.31,1.23,1.18,  # Lu-
    1.16,1.11,1.12,1.13,1.32,  # -Hg
    1.30,1.30,1.36,1.31,1.38,1.42,  # Tl-Rn
    2.01,1.81,  # Fr,Ra
    1.67,1.58,1.52,1.53,1.54,1.55,1.49,  # Ac-Am
    1.49,1.51,1.51,1.48,1.50,1.56,1.58,  # Cm-No
    1.45,1.41,1.34,1.29,1.27,  # Lr-
    1.21,1.16,1.15,1.09,1.22,  # -Cn
    1.36,1.43,1.46,1.58,1.48,1.57  # Nh-Og
])
"""
Covalent radii (taken from Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197).
Values for metals decreased by 10 %.
"""

COV_D3 = 4.0 / 3.0 * COV_2009
"""D3 covalent radii used to construct the coordination number"""
