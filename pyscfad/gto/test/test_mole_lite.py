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

from functools import partial
import pytest
import numpy
import jax
import pyscf
from pyscfad import numpy as np
from pyscfad.gto import Mole, MoleLite

@pytest.fixture
def atom():
    yield "h1 0 0 0; h2 0 0 2"

@pytest.fixture
def basis():
    yield {"H1" : "sto3g", "H2" : "631G**"}

@pytest.fixture
def unit():
    yield "AU"

def int1e_norm(mol, intor="int1e_ovlp"):
    ints = mol.intor(intor,
                     shls_slice=(0, 1, 2, 3),
                     hermi=0)
    return np.linalg.norm(ints)

def int1e_norm1(coords, basis, intor="int1e_ovlp"):
    mol = MoleLite(
        symbols=("h1","h2"),
        coords=coords,
        basis=basis,
        trace_coords=True,
    )
    ints = mol.intor(intor,
                     shls_slice=(0, 1, 2, 3),
                     hermi=0)
    return np.linalg.norm(ints)

def test_int1e(atom, basis, unit):
    coords = np.array([[0,0,0],[0,0,2]], dtype=float)
    mol = Mole()
    mol.atom = atom
    mol.basis = basis
    mol.unit = unit
    mol.build(trace_exp=False, trace_ctr_coeff=False)

    for intor in ["int1e_ovlp", "int1e_kin"]:
        fn = partial(int1e_norm, intor=intor)
        fn1 = lambda x: int1e_norm1(x, basis, intor=intor)
        assert abs(fn1(coords) - fn(mol)) < 1e-8

        gfn = jax.grad(fn)
        gfn1 = jax.jit(jax.grad(fn1))
        grad = gfn(mol).coords
        grad1 = gfn1(coords)
        assert abs(grad1 - grad).max() < 1e-8

        hfn = jax.jacfwd(jax.grad(fn))
        hfn1 = jax.jit(jax.jacfwd(jax.grad(fn1)))
        hess = hfn(mol).coords.coords
        hess1 = hfn1(coords)
        assert abs(hess1 - hess).max() < 1e-8

def test_int1e_origin(atom, basis, unit):
    mol = pyscf.M(atom=atom, basis=basis, unit=unit)
    symbols = tuple(mol.atom_symbol(ia) for ia in range(mol.natm))

    def int_fn(coords, R0, intor, hermi=0, shls_slice=None, origin="common"):
        mol1 = MoleLite(
            symbols=symbols,
            coords=coords,
            basis=basis,
            trace_coords=True,
        )

        if origin == "rinv":
            with mol1.with_rinv_origin(R0):
                int1 = mol1.intor(intor, hermi=hermi, shls_slice=shls_slice)
        elif origin == "common":
            with mol1.with_common_origin(R0):
                int1 = mol1.intor(intor, hermi=hermi, shls_slice=shls_slice)
        else:
            raise NotImplementedError
        return int1

    coords = np.asarray(mol.atom_coords())
    R0 = numpy.array([1., 0., 1.])
    shls_slice = None

    intor_dr01 = {
        "int1e_r": "int1e_irp",
        "int1e_rr": "int1e_irrp",
    }
    for intor in ["int1e_r", "int1e_rr",]:
        with mol.with_common_origin(R0):
            int0 = mol.intor(intor, hermi=1, shls_slice=shls_slice)
            int0_dr01 = mol.intor(intor_dr01[intor], shls_slice=shls_slice)
            int0_dR0 = int0_dr01 + int0_dr01.transpose(0,2,1)
            int0_dR0 = int0_dR0.reshape(-1,3,mol.nao,mol.nao).transpose(0,2,3,1)

            jac = [numpy.zeros_like(int0_dr01) for ia in range(mol.natm)]
            jac = numpy.asarray(jac).reshape(mol.natm,3,-1,mol.nao,mol.nao)
            aoslices = mol.aoslice_by_atom()
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia,2:]
                jac[ia,...,p0:p1] = -int0_dr01[...,p0:p1].reshape(-1,3,mol.nao,p1-p0).transpose(1,0,2,3)
            int0_dR = jac.transpose(2,3,4,0,1)
            int0_dR += int0_dR.transpose(0,2,1,3,4)

        for hermi in (0, 1):
            int1 = int_fn(coords, R0, intor, hermi=hermi, shls_slice=shls_slice)
            int1_dR0 = jax.jacrev(int_fn, 1)(coords, R0, intor, hermi=hermi, shls_slice=shls_slice)
            int1_dR = jax.jacrev(int_fn, 0)(coords, R0, intor, hermi=hermi, shls_slice=shls_slice)
            assert abs(int1 - int0).max() < 1e-8
            assert abs(int1_dR0 - int0_dR0).max() < 1e-8
            assert abs(int1_dR - int0_dR).max() < 1e-8

    with mol.with_rinv_origin(R0):
        int0 = mol.intor("int1e_rinv", hermi=1, shls_slice=shls_slice)
        int0_dr10 = mol.intor("int1e_iprinv", shls_slice=shls_slice)
        int0_dR0 = int0_dr10 + int0_dr10.transpose(0,2,1)
        int0_dR0 = int0_dR0.transpose(1,2,0)

        jac = [numpy.zeros_like(int0_dr10) for ia in range(mol.natm)]
        jac = numpy.asarray(jac).reshape(mol.natm,3,mol.nao,mol.nao)
        aoslices = mol.aoslice_by_atom()
        for ia in range(mol.natm):
            p0, p1 = aoslices[ia,2:]
            jac[ia,:,p0:p1,:] = -int0_dr10[:,p0:p1,:]
        int0_dR = jac.transpose(2,3,0,1)
        int0_dR += int0_dR.transpose(1,0,2,3)

        for hermi in (0, 1):
            int1 = int_fn(coords, R0, "int1e_rinv", hermi=hermi, shls_slice=shls_slice,
                          origin="rinv")
            int1_dR0 = jax.jacrev(int_fn, 1)(coords, R0, "int1e_rinv", hermi=hermi,
                                             shls_slice=shls_slice, origin="rinv")
            int1_dR = jax.jacrev(int_fn, 0)(coords, R0, "int1e_rinv", hermi=hermi,
                                            shls_slice=shls_slice, origin="rinv")
            assert abs(int1 - int0).max() < 1e-8
            assert abs(int1_dR0 - int0_dR0).max() < 1e-8
            assert abs(int1_dR - int0_dR).max() < 1e-8

def test_from_to_pyscf(atom, basis, unit):
    pmol = pyscf.M(atom=atom, basis=basis, unit=unit)
    mol = MoleLite.from_pyscf(pmol)
    pmol1 = mol.to_pyscf()

    assert jax.tree.all(jax.tree.map(np.allclose, pmol._basis, pmol1._basis))
