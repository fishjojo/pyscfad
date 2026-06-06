# Copyright 2021-2025 The PySCFAD Authors
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

from __future__ import annotations
from abc import ABC, abstractmethod
import dataclasses

import numpy
import jax

from pyscf.data.elements import _symbol
from pyscf.data.nist import HARTREE2EV

from pyscfad.typing import Array
from pyscfad import numpy as np
from pyscfad.xtb.param import GFN1Param, cn_d3
from pyscfad.xtb.util import atom_to_bas_indices, atom_to_bas_indices_2d, ANG_MOMENT

from pyscfad.ml.gto import MolePad, BasisArray


def make_gfn1_param_array(
    basis: BasisArray,
    max_number: int,
) -> GFN1ParamArray:
    param = GFN1Param()

    kf = param.kf
    kcn_d3 = param.kcn_d3
    kEN = param.kEN

    EN = numpy.zeros((max_number+1,), dtype=np.floatx)
    zeff = numpy.zeros((max_number+1,), dtype=np.floatx)
    arep = numpy.zeros((max_number+1,), dtype=np.floatx)
    refocc = numpy.zeros((max_number+1, basis.nbas), dtype=np.floatx)
    gam = numpy.ones((max_number+1,), dtype=np.floatx) # use ones to avoid gradient divergence
    lgam = numpy.ones((max_number+1, basis.nbas), dtype=np.floatx) # use ones to avoid gradient divergence
    gam3 = numpy.zeros((max_number+1,), dtype=np.floatx)
    selfenergy = numpy.zeros((max_number+1, basis.nbas), dtype=np.floatx)
    shpoly = numpy.zeros((max_number+1, basis.nbas), dtype=np.floatx)
    kcn = numpy.zeros((max_number+1, basis.nbas), dtype=np.floatx)
    kpair = numpy.ones((max_number+1, max_number+1), dtype=np.floatx)
    for z in range(1,max_number+1):
        symb = _symbol(z)
        EN[z] = param.element[symb].en
        zeff[z] = param.element[symb].zeff
        arep[z] = param.element[symb].arep
        refocc[z][basis.mask_shl[z]] = param.element[symb].refocc
        gam[z] = param.element[symb].gam
        lgam[z][basis.mask_shl[z]] = param.element[symb].lgam
        gam3[z] = param.element[symb].gam3
        selfenergy[z][basis.mask_shl[z]] = param.element[symb].levels / HARTREE2EV
        shpoly[z][basis.mask_shl[z]] = param.element[symb].shpoly
        kcn[z][basis.mask_shl[z]] = param.element[symb].kcn / HARTREE2EV

        for z1 in range(1, z+1):
            key = f"{_symbol(z)}-{_symbol(z1)}"
            kpair[z,z1] = param.kpair.get(key, 1)
            if z != z1:
                kpair[z1,z] = kpair[z,z1]

    k_shlpr = numpy.ones((basis.nbas, basis.nbas), dtype=np.floatx)
    ls = basis.ls.copy()
    if numpy.bincount(ls)[0] == 2:
        ls[1] = -20 # H 2s
    for i, li in enumerate(ls):
        for j, lj in enumerate(ls):
            key = ANG_MOMENT[min(li, lj)] + ANG_MOMENT[max(li, lj)]
            k_shlpr[i,j] = param.k_shlpr.get(key, 1)

    return GFN1ParamArray(kf=kf,
                          kcn_d3=kcn_d3,
                          kEN=kEN,
                          EN=np.asarray(EN),
                          zeff=np.asarray(zeff),
                          arep=np.asarray(arep),
                          refocc=np.asarray(refocc),
                          gam=np.asarray(gam),
                          lgam=np.asarray(lgam),
                          gam3=np.asarray(gam3),
                          selfenergy=np.asarray(selfenergy),
                          shpoly=np.asarray(shpoly),
                          kcn=np.asarray(kcn),
                          kpair=np.asarray(kpair),
                          k_shlpr=np.asarray(k_shlpr))

def make_param_array(
    basis: BasisArray,
    max_number: int,
    dataset: str = "GFN1",
) -> ParamArray:
    if dataset.upper() == "GFN1":
        return make_gfn1_param_array(basis, max_number)
    else:
        raise NotImplementedError

class ParamArray(ABC):
    @abstractmethod
    def to_mol_param(self, mol: MolePad) -> MoleParam:
        raise NotImplementedError

@jax.tree_util.register_dataclass
@dataclasses.dataclass
class GFN1ParamArray(ParamArray):
    kf: float
    kcn_d3: float
    kEN: float
    EN: Array
    zeff: Array
    arep: Array
    refocc: Array
    gam: Array
    lgam: Array
    gam3: Array
    selfenergy: Array
    shpoly: Array
    kcn: Array
    kpair: Array
    k_shlpr: Array
    dipgam: Array | None = None
    quadgam: Array | None = None

    def to_mol_param(self, mol: MolePad) -> GFN1MoleParam:
        kf = self.kf
        kEN = self.kEN
        EN = self.EN[mol.numbers]
        zeff = self.zeff[mol.numbers]
        arep = self.arep[mol.numbers]
        refocc = self.refocc[mol.numbers].ravel()
        gam = self.gam[mol.numbers][atom_to_bas_indices(mol)]
        if self.dipgam is not None:
            dipgam = self.dipgam[mol.numbers]
        else:
            dipgam = None
        if self.quadgam is not None:
            quadgam = self.quadgam[mol.numbers]
        else:
            quadgam = None
        lgam = self.lgam[mol.numbers].ravel()
        gam3 = self.gam3[mol.numbers]
        selfenergy = self.selfenergy[mol.numbers].ravel()
        shpoly = self.shpoly[mol.numbers].ravel()
        kcn = self.kcn[mol.numbers].ravel()
        CN = cn_d3(mol, kcn=self.kcn_d3)
        kpair = self.kpair[np.ix_(mol.numbers, mol.numbers)]
        kpair = kpair[atom_to_bas_indices_2d(mol)]

        l_to_bas_id = np.tile(np.arange(mol.basis.nbas), mol.natm)
        k_shlpr = self.k_shlpr[np.ix_(l_to_bas_id, l_to_bas_id)]
        mask = np.outer(mol.shl_mask, mol.shl_mask)
        k_shlpr = np.where(mask, k_shlpr, 1)
        return GFN1MoleParam(kf=kf,
                             kEN=kEN,
                             EN=EN,
                             zeff=zeff,
                             arep=arep,
                             refocc=refocc,
                             gam=gam,
                             lgam=lgam,
                             gam3=gam3,
                             selfenergy=selfenergy,
                             shpoly=shpoly,
                             kcn=kcn,
                             CN=CN,
                             kpair=kpair,
                             k_shlpr=k_shlpr,
                             dipgam=dipgam,
                             quadgam=quadgam,
                             )

class MoleParam(ABC):
    pass

@jax.tree_util.register_dataclass
@dataclasses.dataclass
class GFN1MoleParam(MoleParam):
    kf: float
    kEN: float
    EN: Array
    zeff: Array
    arep: Array
    refocc: Array
    gam: Array
    lgam: Array
    gam3: Array
    selfenergy: Array
    shpoly: Array
    kcn: Array
    CN: Array
    kpair: Array
    k_shlpr: Array
    dipgam: Array | None = None
    quadgam: Array | None = None

