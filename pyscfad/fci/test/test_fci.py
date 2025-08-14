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

from functools import reduce
import pytest
import jax
from jax import numpy as np
from pyscfad import gto, scf, ao2mo
from pyscfad.fci import fci_slow

@pytest.fixture()
def get_h2():
    mol = gto.Mole()
    mol.atom    = 'H 0 0 0; H 0 0 0.74'
    mol.basis   = 'sto3g'
    mol.verbose = 0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol

def fci_energy(mol):
    mf = scf.RHF(mol)
    mf.kernel()
    norb = mf.mo_coeff.shape[1]
    nelec = mol.nelectron
    h1e = reduce(np.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
    eri = ao2mo.incore.full(mf._eri, mf.mo_coeff, compact=False)
    eri = eri.reshape(norb,norb,norb,norb)

    e1 = fci_slow.kernel(h1e, eri, norb, nelec, mf.energy_nuc())[0]
    return e1

def test_nuc_grad(get_h2):
    mol = get_h2
    e = fci_energy(mol)
    assert abs(e - -1.1372838344885026) < 1e-9

    g = jax.grad(fci_energy)(mol).coords
    assert abs(g[0,2] - -0.00455429) < 1e-6
    assert abs(g.sum()) < 1e-6
