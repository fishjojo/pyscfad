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
from pyscf.pbc.lib.kpts_helper import is_zero, member
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.ops import vmap
from pyscfad.pbc import tools

def _ewald_exxdiv_for_G0(cell, kpts, dms, vk, kpts_band=None):
    s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
    s = np.asarray(s)
    madelung = tools.pbc.madelung(cell, kpts)

    def _vk_corr(dm, s):
        return np.dot(s, np.dot(dm, s)) * madelung

    if kpts is None:
        vk += vmap(_vk_corr, in_axes=(0, None))(dms, s)

    elif numpy.shape(kpts) == (3,):
        if kpts_band is None or is_zero(kpts_band-kpts):
            vk += vmap(_vk_corr, in_axes=(0, None))(dms, s)

    elif kpts_band is None or numpy.array_equal(kpts, kpts_band):
        for i, dm in enumerate(dms):
            vk = ops.index_add(vk, ops.index[i], vmap(_vk_corr)(dm, s))

    else:
        for k, kpt in enumerate(kpts):
            for kp in member(kpt, kpts_band.reshape(-1,3)):
                for i,dm in enumerate(dms):
                    #vk[i,kp] += madelung * reduce(numpy.dot, (s[k], dm[k], s[k]))
                    vk = ops.index_add(vk, ops.index[i,kp],
                                   madelung * np.dot(s[k], np.dot(dm[k], s[k])))
    return vk
