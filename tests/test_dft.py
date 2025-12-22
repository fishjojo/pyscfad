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

"""Tests for pyscfad.dft
"""
import numpy
import jax
from pyscf import dft as pyscf_dft
from pyscfad import config_update
from pyscfad import dft
from .util import (
    dft_energy,
    df_dft_energy,
    dft_nuc_grad,
    dft_nuc_hess,
)

LDA = ["lda,vwn",]
GGA = ["pbe,pbe",]
HYB = ["b3lyp",]
LRC = ["HYB_GGA_XC_LRC_WPBE",]
MGGA = ["m062x",]
NLC = ["B97M_V",]

def test_rks_nuc_grad(mol_H2):
    mol = mol_H2()
    for xc in LDA + GGA + HYB + LRC + MGGA + NLC:
        g1 = jax.grad(dft_energy)(mol, dft.RKS, xc).coords
        g0 = dft_nuc_grad(mol, dft.RKS, xc)
        assert abs(g1 - g0).max() < 1e-6

def test_uks_nuc_grad(mol_H2):
    mol = mol_H2()
    for xc in LDA + GGA + HYB + LRC: # FIXME mgga
        g1 = jax.grad(dft_energy)(mol, dft.UKS, xc).coords
        g0 = dft_nuc_grad(mol, dft.UKS, xc)
        assert abs(g1 - g0).max() < 1e-6

def test_df_rks_nuc_grad(mol_H2):
    mol = mol_H2()
    g0 = numpy.array([[0,0,0.00738732], [0,0,-0.00738732]])
    with config_update('pyscfad_scf_implicit_diff', True):
        g1 = jax.grad(df_dft_energy)(mol, dft.RKS, "pbe,pbe").coords
    assert abs(g1 - g0).max() < 1e-6

def test_df_uks_nuc_grad(mol_H2):
    mol = mol_H2()
    g0 = numpy.array([[0,0,0.00738732], [0,0,-0.00738732]])
    with config_update('pyscfad_scf_implicit_diff', True):
        g1 = jax.grad(df_dft_energy)(mol, dft.UKS, "pbe,pbe").coords
    assert abs(g1 - g0).max() < 1e-6

def test_rks_nuc_hess(mol_H2):
    mol = mol_H2()
    for xc in LDA + GGA + HYB:
        hess1 = jax.hessian(dft_energy)(mol, dft.RKS, xc).coords.coords
        hess0 = dft_nuc_hess(mol, pyscf_dft.RKS, xc)
        assert abs(hess1 - hess0).max() < 1e-6

def test_uks_nuc_hess(mol_H2):
    mol = mol_H2()
    for xc in LDA + GGA + HYB:
        hess1 = jax.hessian(dft_energy)(mol, dft.UKS, xc).coords.coords
        hess0 = dft_nuc_hess(mol, pyscf_dft.UKS, xc)
        assert abs(hess1 - hess0).max() < 1e-6
