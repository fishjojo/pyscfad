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
from pyscfad import numpy as np
from pyscfad import ops

def get_gth_vlocG_part1(cell, Gv):
    from pyscfad.pbc import tools
    coulG = tools.get_coulG(cell, Gv=Gv)
    G2 = np.einsum('ix,ix->i', Gv, Gv)
    G0idx = np.where(G2==0)[0]

    if cell.dimension != 2 or cell.low_dim_ft_type == 'inf_vacuum':
        vlocG = np.zeros((cell.natm, len(G2)))
        for ia in range(cell.natm):
            Zia = cell.atom_charge(ia)
            symb = cell.atom_symbol(ia)
            # Note the signs -- potential here is positive
            vlocG = ops.index_update(vlocG, ops.index[ia], Zia * coulG)
            if symb in cell._pseudo:
                pp = cell._pseudo[symb]
                rloc, nexp, cexp = pp[1:3+1]
                vlocG = ops.index_mul(vlocG, ops.index[ia], np.exp(-0.5*rloc**2 * G2))
                # alpha parameters from the non-divergent Hartree+Vloc G=0 term.
                vlocG = ops.index_update(vlocG, ops.index[ia,G0idx], -2*numpy.pi*Zia*rloc**2)

    elif cell.dimension == 2:
        raise NotImplementedError
    else:
        raise NotImplementedError(f'Low dimension ft_type {cell.low_dim_ft_type}'
                                  f' not implemented for dimension {cell.dimension}')
    return vlocG
