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

import jax
from pyscfad import scf
from pyscfad import config_update

def test_df_hf_nuc_grad(get_h2):
    mol = get_h2
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    dm = mf.make_rdm1()
    def hf_energy(mf, dm0=None):
        mf.reset() # reset to get with_df's gradient
        e_tot = mf.kernel(dm0=dm0)
        return e_tot
    jac = jax.grad(hf_energy)(mf)
    g = jac.mol.coords + jac.with_df.mol.coords+jac.with_df.auxmol.coords

    jac = jax.grad(hf_energy)(mf, dm)
    g1 = jac.mol.coords + jac.with_df.mol.coords+jac.with_df.auxmol.coords
    assert abs(g1 - g).max() < 1e-6
    assert abs(g1[1,2] - 0.007562137919067152) < 1e-6 # finite difference reference

def test_df_hf_nuc_grad1(get_h2):
    with config_update('pyscfad_scf_implicit_diff', True):
        mol = get_h2
        def energy(mol):
            mf = scf.RHF(mol).density_fit()
            return mf.kernel()
        g = jax.grad(energy)(mol).coords
        assert abs(g[1,2] - 0.007562137919067152) < 1e-6

def test_to_pyscf(get_n2):
    mol = get_n2
    mf = scf.UHF(mol).density_fit().to_pyscf()
    ehf = mf.kernel()
    assert abs(ehf - -108.867850114325) < 1e-8
