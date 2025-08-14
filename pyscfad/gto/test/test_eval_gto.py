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
import numpy as np
import jax
import pyscf
from pyscf.dft import gen_grid
from pyscf.gto.eval_gto import eval_gto as pyscf_eval_gto
from pyscfad import gto

BOHR = 0.52917721092
bas = 'sto3g'
eval_names = ["GTOval_sph", "GTOval_sph_deriv1",
              "GTOval_sph_deriv2", "GTOval_sph_deriv3",]

def test_eval_gto_nuc():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74'  # in Angstrom
    mol.basis = bas
    mol.build(trace_coords=True, trace_exp=False, trace_ctr_coeff=False)

    grids = gen_grid.Grids(mol)
    grids.atom_grid = {'H': (10, 14)}
    grids.build(with_non0tab=True)
    coords = grids.coords

    molp = pyscf.gto.Mole()
    molp.atom = 'H 0 0 0; H 0 0 0.740005'  # in Angstrom
    molp.basis = bas
    molp.build()

    molm = pyscf.gto.Mole()
    molm.atom = 'H 0 0 0; H 0 0 0.739995'  # in Angstrom
    molm.basis = bas
    molm.build()

    tol = [1e-6,] * 4

    for i, eval_name in enumerate(eval_names):
        ao0 = pyscf_eval_gto(mol, eval_name, coords)
        ao = mol.eval_gto(eval_name, coords)
        assert abs(ao-ao0).max() < 1e-10

        aop = molp.eval_gto(eval_name, coords)
        aom = molm.eval_gto(eval_name, coords)
        g_fd = (aop-aom) / (1e-5 / BOHR)
        jac_fwd = jax.jacfwd(mol.__class__.eval_gto)(mol, eval_name, coords)
        assert abs(jac_fwd.coords[...,1,2] - g_fd).max() < tol[i]

        #jac_rev = jax.jacrev(mol.__class__.eval_gto)(mol, eval_name, coords)
        #assert abs(jac_rev.coords[...,1,2] - g_fd).max() < tol[i]

def four_point_fd(mol, eval_name, coords, _env_of, disp=1e-4):
    grad_fd = []
    for _, ptr_exp in enumerate(_env_of):
        #ptr_exp = _env_of[i]
        mol._env[ptr_exp] += disp
        sp = mol.eval_gto(eval_name, coords)
        mol._env[ptr_exp] += disp
        sp2 = mol.eval_gto(eval_name, coords)
        mol._env[ptr_exp] -= disp * 4.
        sm2 = mol.eval_gto(eval_name, coords)
        mol._env[ptr_exp] += disp
        sm = mol.eval_gto(eval_name, coords)
        g = (8.*(sp-sm) - sp2 + sm2) / (12.*disp)
        grad_fd.append(g)
        mol._env[ptr_exp] += disp
    grad_fd = np.asarray(grad_fd)
    grad_fd = np.moveaxis(grad_fd, 0, -1)
    return grad_fd

def cs_grad_fd(mol, eval_name, coords):
    disp = 1e-3
    _, _, _env_of = gto.mole.setup_ctr_coeff(mol)
    g = four_point_fd(mol, eval_name, coords, _env_of, disp)
    return g

def exp_grad_fd(mol, eval_name, coords):
    disp = 1e-4
    _, _, _env_of = gto.mole.setup_exp(mol)
    g = four_point_fd(mol, eval_name, coords, _env_of, disp)
    return g

def test_eval_gto_cs():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; Li 0 0 1.6'  # in Angstrom
    mol.basis = bas
    mol.build(trace_coords=False, trace_ctr_coeff=True, trace_exp=False)

    grids = gen_grid.Grids(mol)
    grids.atom_grid = {'H': (10, 14), 'Li': (15, 14)}
    grids.build(with_non0tab=True)
    coords = grids.coords

    tol = [1e-6,] * 4

    for i, eval_name in enumerate(eval_names):
        g_fd = cs_grad_fd(mol, eval_name, coords)

        jac_fwd = jax.jacfwd(mol.__class__.eval_gto)(mol, eval_name, coords)
        assert abs(jac_fwd.ctr_coeff - g_fd).max() < tol[i]

        #jac_rev = jax.jacrev(mol.__class__.eval_gto)(mol, eval_name, coords)
        #assert abs(jac_rev.ctr_coeff - g_fd).max() < tol[i]

def test_eval_gto_exp():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; Li 0 0 1.6'  # in Angstrom
    mol.basis = bas
    mol.build(trace_coords=False, trace_ctr_coeff=False, trace_exp=True)

    grids = gen_grid.Grids(mol)
    grids.atom_grid = {'H': (10, 14), 'Li': (15, 14)}
    grids.build(with_non0tab=True)
    coords = grids.coords

    tol = [1e-6,] * 4

    for i, eval_name in enumerate(eval_names):
        g_fd = exp_grad_fd(mol, eval_name, coords)

        jac_fwd = jax.jacfwd(mol.__class__.eval_gto)(mol, eval_name, coords)
        assert abs(jac_fwd.exp - g_fd).max() < tol[i]

        #jac_rev = jax.jacrev(mol.__class__.eval_gto)(mol, eval_name, coords)
        #assert abs(jac_rev.exp - g_fd).max() < tol[i]
