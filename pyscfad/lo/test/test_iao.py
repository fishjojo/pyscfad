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

import pytest
import jax
from pyscf.data.nist import BOHR
from pyscfad import gto, scf
from pyscfad.lo import iao, orth
from pyscfad import config

@pytest.fixture
def get_mol():
    config.update('pyscfad_scf_implicit_diff', True)
    #config.update('pyscfad_moleintor_opt', True)

    mol = gto.Mole()
    mol.atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587'
    mol.basis = '631G'
    mol.verbose = 0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    yield mol

    config.reset()

def _iao(mol):
    mf = scf.RHF(mol)
    mf.kernel()
    orbocc = mf.mo_coeff[:,mf.mo_occ>1e-6]
    c = iao.iao(mol, orbocc)
    c = orth.vec_lowdin(c, mf.get_ovlp())
    return c

def test_iao(get_mol):
    mol = get_mol
    jac = jax.jacrev(_iao)(mol)
    g0 = jac.coords[:,:,0,2]

    mol.set_geom_('O 0. 0.  0.001; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587')
    c1 = _iao(mol)

    mol.set_geom_('O 0. 0. -0.001; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587')
    c2 = _iao(mol)

    g1 = (c1 - c2) / (0.002 / BOHR)
    assert abs(g0-g1).max() < 1e-6
