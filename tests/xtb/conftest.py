# Copyright 2025-2026 The PySCFAD Authors
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

import pytest
from pyscfad import numpy as np

@pytest.fixture
def H2O_GFN1_ref():
    numbers = np.array([8, 1, 1])
    coords = np.array(
        [
            [0.00000,  0.00000,  0.00000],
            [1.43355,  0.00000, -0.95296],
            [1.43355,  0.00000,  0.95296],
        ]
    )
    e = -5.7231173051309865
    g = np.array(
        [
            [ 3.93688356e-02, 0.,  0.],
            [-1.96844178e-02, 0.,  1.03173514e-01],
            [-1.96844178e-02, 0., -1.03173514e-01],
        ]
    )
    mu = np.array([3.77507247, 0., 0.])
    alpha = np.diag(np.array([7.33970468, 1.07102937e-01, 4.97202824]))
    yield (numbers, coords, e, g, mu, alpha)

@pytest.fixture
def NH3_GFN1_ref():
    numbers = np.array([7, 1, 1, 1])
    coords = np.array(
        [
            [-0.80650, -1.00659,  0.02850],
            [-0.50540, -0.31299,  0.68220],
            [ 0.00620, -1.41579, -0.38500],
            [-1.32340, -0.54779, -0.69350],
        ]
    ) / 0.52917721067121
    e = -4.82989868
    g = np.array(
        [
            [ 0.00307771,  0.00378012, -0.00250677],
            [-0.00197155, -0.00543102, -0.00679491],
            [-0.00681684,  0.00491176,  0.0032046 ],
            [ 0.00571069, -0.00326086,  0.00609709],
        ]
    )
    mu = np.array([1.14764393,  1.42879716, -0.92636369])
    alpha = np.array(
        [
            [ 8.40057146, -3.05976767,  1.98425083],
            [-3.05976767,  7.04997063,  2.47067676],
            [ 1.98425083,  2.47067676,  9.25966469],
        ]
    )
    yield (numbers, coords, e, g, mu, alpha)
