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
from pyscfad import gto, dft
from pyscfad import config_update

BOHR = 0.52917721092
disp = 1e-4

# pylint: disable=redefined-outer-name
def test_rks_nuc_grad_lda(get_mol):
    mol = get_mol
    mf = dft.RKS(mol)
    mf.xc = 'lda,vwn'
    g1 = mf.energy_grad(mode="rev").coords
    mf.kernel()
    g2 = mf.energy_grad(mode="rev").coords
    g0 = mf.nuc_grad_method().kernel()
    assert abs(g1-g0).max() < 1e-6
    assert abs(g2-g0).max() < 1e-6

def test_rks_nuc_grad_gga(get_mol):
    mol = get_mol()
    mf = dft.RKS(mol)
    mf.xc = 'pbe,pbe'
    g1 = mf.energy_grad(mode="rev").coords
    mf.kernel()
    g2 = mf.energy_grad(mode="rev").coords
    g0 = mf.nuc_grad_method().kernel()
    assert abs(g1-g0).max() < 1e-6
    assert abs(g2-g0).max() < 1e-6

def test_rks_nuc_grad_hybrid(get_mol):
    mol = get_mol()
    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    g1 = mf.energy_grad(mode="rev").coords
    mf.kernel()
    g2 = mf.energy_grad(mode="rev").coords
    g0 = mf.nuc_grad_method().kernel()
    assert abs(g1-g0).max() < 1e-6
    assert abs(g2-g0).max() < 1e-6

def test_rks_nuc_grad_lrc(get_mol):
    mol = get_mol()
    mf = dft.RKS(mol)
    mf.xc = 'HYB_GGA_XC_LRC_WPBE'
    g1 = mf.energy_grad(mode="rev").coords
    mf.kernel()
    g2 = mf.energy_grad(mode="rev").coords
    g0 = mf.nuc_grad_method().kernel()
    assert abs(g1-g0).max() < 1e-6
    assert abs(g2-g0).max() < 1e-6

def test_rks_nuc_grad_mgga(get_mol, get_mol_p, get_mol_m):
    mol = get_mol
    mf = dft.RKS(mol)
    mf.xc = 'm062x'
    g1 = mf.energy_grad(mode="rev").coords
    mf.kernel()
    g2 = mf.energy_grad(mode="rev").coords
    assert abs(g1 - g2).max() < 1e-5

    molp = get_mol_p
    mfp = dft.RKS(molp)
    mfp.xc = 'm062x'
    ep = mfp.kernel()

    molm = get_mol_m
    mfm = dft.RKS(molm)
    mfm.xc = 'm062x'
    em = mfm.kernel()

    g_fd = (ep-em) / disp * BOHR
    assert abs(g2[1,2] - g_fd) < 1e-5

def test_rks_nuc_grad_nlc(get_mol, get_mol_p, get_mol_m):
    mol = get_mol
    mf = dft.RKS(mol)
    mf.xc = 'B97M_V'
    mf.nlc = 'VV10'
    g = mf.energy_grad(mode="rev").coords

    molp = get_mol_p
    mfp = dft.RKS(molp)
    mfp.xc = 'B97M_V'
    mfp.nlc = 'VV10'
    ep = mfp.kernel()

    molm = get_mol_m
    mfm = dft.RKS(molm)
    mfm.xc = 'B97M_V'
    mfm.nlc = 'VV10'
    em = mfm.kernel()

    g_fd = (ep-em) / disp * BOHR
    assert abs(g[1,2] - g_fd) < 1e-5

def test_df_rks_nuc_grad(get_mol):
    with config_update('pyscfad_scf_implicit_diff', True):
        mol = get_mol
        def energy(mol):
            mf = dft.RKS(mol, xc='PBE,PBE').density_fit()
            return mf.kernel()
        g = jax.grad(energy)(mol).coords
        # finite difference reference
        assert abs(g[1,2] - -0.007387325326884536) < 1e-6
