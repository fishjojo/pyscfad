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

from functools import wraps
import numpy
from pyscf.pbc.gto.pseudo import pp as pyscf_pp

from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.pbc.gto.pseudo import pp_int

@wraps(pyscf_pp.get_vlocG)
def get_vlocG(cell, Gv=None):
    if Gv is None:
        Gv = cell.Gv
    vlocG = get_gth_vlocG(cell, Gv)
    return vlocG

@wraps(pyscf_pp.get_gth_vlocG)
def get_gth_vlocG(cell, Gv):
    vlocG = pp_int.get_gth_vlocG_part1(cell, Gv)

    # Add the C1, C2, C3, C4 contributions
    G2 = np.einsum("ix,ix->i", Gv, Gv)
    for ia in range(cell.natm):
        symb = cell.atom_symbol(ia)
        if symb not in cell._pseudo:
            continue

        pp = cell._pseudo[symb]
        rloc, nexp, cexp = pp[1:3+1]

        G2_red = G2 * rloc**2
        cfacs = 0
        if nexp >= 1:
            cfacs += cexp[0]
        if nexp >= 2:
            cfacs += cexp[1] * (3 - G2_red)
        if nexp >= 3:
            cfacs += cexp[2] * (15 - 10*G2_red + G2_red**2)
        if nexp >= 4:
            cfacs += cexp[3] * (105 - 105*G2_red + 21*G2_red**2 - G2_red**3)

        vlocG = ops.index_add(vlocG, ops.index[ia,:],
                              -(2*numpy.pi)**(3/2.)*rloc**3*np.exp(-0.5*G2_red) * cfacs)

    return vlocG

def _qli(x,l,i):
    sqrt = np.sqrt
    if l==0 and i==0:
        return 4*sqrt(2.)
    elif l==0 and i==1:
        return 8*sqrt(2/15.)*(3-x**2) # MH & GTH (right)
        #return sqrt(8*2/15.)*(3-x**2) # HGH (wrong)
    elif l==0 and i==2:
        #return 16/3.*sqrt(2/105.)*(15-20*x**2+4*x**4) # MH (wrong)
        return 16/3.*sqrt(2/105.)*(15-10*x**2+x**4) # HGH (right)
    elif l==1 and i==0:
        return 8*sqrt(1/3.)
    elif l==1 and i==1:
        return 16*sqrt(1/105.)*(5-x**2)
    elif l==1 and i==2:
        #return 32/3.*sqrt(1/1155.)*(35-28*x**2+4*x**4) # MH (wrong)
        return 32/3.*sqrt(1/1155.)*(35-14*x**2+x**4) # HGH (right)
    elif l==2 and i==0:
        return 8*sqrt(2/15.)
    elif l==2 and i==1:
        return 16/3.*sqrt(2/105.)*(7-x**2)
    elif l==2 and i==2:
        #return 32/3.*sqrt(2/15015.)*(63-36*x**2+4*x**4) # MH (wrong I think)
        return 32/3.*sqrt(2/15015.)*(63-18*x**2+x**4) # TCB
    elif l==3 and i==0:
        return 16*sqrt(1/105.)
    elif l==3 and i==1:
        return 32/3.*sqrt(1/1155.)*(9-x**2)
    elif l==3 and i==2:
        return 64/45.*sqrt(1/1001.)*(99-22*x**2+x**4)
    elif l==4 and i==0:
        return 16/3.*sqrt(2/105.)
    elif l==4 and i==1:
        return 32/3.*sqrt(2/15015.)*(11-x**2)
    elif l==4 and i==2:
        return 64/45.*sqrt(2/17017.)*(143-26*x**2+x**4)
    else:
        print("*** WARNING *** l =", l, ", i =", i, "not yet implemented for NL PP!")
        return 0.
