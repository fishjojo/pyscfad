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
from pyscfad.gto._pyscf_moleintor import _INTOR_FUNCTIONS

def rng_tangent(shape, seed=0):
    return numpy.random.default_rng(seed).standard_normal(shape)

@pytest.fixture(scope="module")
def atom():
    yield "h1 0 0 0; h2 0 0 2"

@pytest.fixture(scope="module")
def basis():
    yield {"H1" : "sto3g", "H2" : "631G**"}

@pytest.fixture(scope="module")
def unit():
    yield "AU"

def int1e_norm(mol, intor="int1e_ovlp", shls_slice=None, hermi=0):
    ints = mol.intor(intor, shls_slice=shls_slice, hermi=hermi)
    return np.linalg.norm(ints)

def int1e_norm1(coords, basis, intor="int1e_ovlp", shls_slice=None, hermi=0):
    mol = MoleLite(symbols=("h1", "h2"), coords=coords, basis=basis)
    ints = mol.intor(intor, shls_slice=shls_slice, hermi=hermi)
    return np.linalg.norm(ints)

def test_int1e(atom, basis, unit):
    coords = np.array([[0,0,0], [0,0,2]], dtype=float)
    mol = Mole()
    mol.atom = atom
    mol.basis = basis
    mol.unit = unit
    mol.build(trace_exp=False, trace_ctr_coeff=False)

    for intor in ["int1e_ovlp", "int1e_kin", "int1e_nuc"]:
        for shls_slice, hermi in zip((None, (0, 1, 2, 3)), (1, 0)):
            fn = partial(int1e_norm, intor=intor,
                         shls_slice=shls_slice, hermi=hermi)
            fn1 = lambda x: int1e_norm1(x, basis, intor=intor,
                                        shls_slice=shls_slice, hermi=hermi)

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
        mol1 = MoleLite(symbols=symbols, coords=coords, basis=basis)

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

def test_origin_array_like(atom, basis, unit):
    # set_common_origin / set_rinv_origin are annotated to accept ArrayLike,
    # so a plain Python sequence must work identically to an ndarray.
    mol = MoleLite(
        symbols=("h1", "h2"),
        coords=np.array([[0., 0., 0.], [0., 0., 2.]]),
        basis=basis,
    )
    R0_list = [1., 0., 1.]
    R0_arr = numpy.asarray(R0_list)

    with mol.with_common_origin(R0_list):
        r_list = mol.intor("int1e_r")
    with mol.with_common_origin(R0_arr):
        r_arr = mol.intor("int1e_r")
    assert abs(r_list - r_arr).max() < 1e-12

    with mol.with_rinv_origin(R0_list):
        rinv_list = mol.intor("int1e_rinv")
    with mol.with_rinv_origin(R0_arr):
        rinv_arr = mol.intor("int1e_rinv")
    assert abs(rinv_list - rinv_arr).max() < 1e-12

def test_from_to_pyscf(atom, basis, unit):
    pmol = pyscf.M(atom=atom, basis=basis, unit=unit)
    mol = MoleLite.from_pyscf(pmol)
    pmol1 = mol.to_pyscf()

    assert jax.tree.all(jax.tree.map(np.allclose, pmol._basis, pmol1._basis))

@pytest.fixture(scope="module")
def coords():
    yield np.array([[0,0,0], [0,0,2]], dtype=float)

@pytest.fixture(scope="module")
def int2e_jac_ref(atom, basis, unit):
    """``d(ij|kl)/dR`` of the unpacked integral of the legacy ``Mole``."""
    mol = Mole()
    mol.atom = atom
    mol.basis = basis
    mol.unit = unit
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    jac = jax.jacfwd(lambda m: m.intor("int2e", aosym="s1"))(mol)
    return numpy.asarray(jac.coords)

@pytest.fixture(scope="module")
def int2e_hess_ref(atom, basis, unit):
    """``d2(ij|kl)/dR2`` of the unpacked integral of the legacy ``Mole``."""
    mol = Mole()
    mol.atom = atom
    mol.basis = basis
    mol.unit = unit
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    hess = jax.jacfwd(jax.jacfwd(lambda m: m.intor("int2e", aosym="s1")))(mol)
    return numpy.asarray(hess.coords.coords)

def int2e_packed(coords, basis, aosym="s4", shls_slice=None, intor="int2e"):
    mol = MoleLite(symbols=("h1", "h2"), coords=coords, basis=basis)
    return mol.intor(intor, aosym=aosym, shls_slice=shls_slice)

def restore(aosym, eri_s1):
    """Pack the index pairs of an 8-fold symmetric ``(ij|kl)`` block,
    keeping any trailing (derivative) axes.
    """
    naoi, _, naok = eri_s1.shape[:3]
    i, j = numpy.tril_indices(naoi)
    k, l = numpy.tril_indices(naok)
    eri = eri_s1[i,j][:,k,l]
    if aosym == "s8":
        # the s8 vector is the lower triangle of the s4 matrix
        p, q = numpy.tril_indices(eri.shape[0])
        eri = eri[p,q]
    return eri

@pytest.mark.parametrize("aosym", ["s4", "s8"])
def test_int2e(coords, basis, atom, unit, int2e_jac_ref, aosym):
    mol0 = pyscf.M(atom=atom, basis=basis, unit=unit)
    fn = partial(int2e_packed, aosym=aosym)

    eri = fn(coords, basis)
    assert abs(eri - mol0.intor("int2e", aosym=aosym)).max() < 1e-10

    jac0 = restore(aosym, int2e_jac_ref)
    jac = numpy.asarray(jax.jacfwd(fn)(coords, basis))
    assert jac.shape == jac0.shape
    assert abs(jac - jac0).max() < 1e-10

    jac_rev = numpy.asarray(jax.jacrev(fn)(coords, basis))
    assert abs(jac_rev - jac0).max() < 1e-10

    norm = lambda x: np.linalg.norm(fn(x, basis))
    grad = numpy.asarray(jax.grad(norm)(coords))
    grad_jit = numpy.asarray(jax.jit(jax.grad(norm))(coords))
    assert abs(grad_jit - grad).max() < 1e-12

# H1 carries a single s shell, H2 an s, s, p sequence
SLICES_2E = [
    (0, 1, 0, 1, 1, 4, 1, 4),  # bra pair on H1, ket pair on H2
    (1, 4, 1, 4, 0, 1, 0, 1),  # the transposed block
    (1, 4, 1, 4, 1, 4, 1, 4),  # bra range = ket range, all indices on H2
]

@pytest.mark.parametrize("shls_slice", SLICES_2E)
def test_int2e_s4_shls_slice(coords, basis, atom, unit, int2e_jac_ref,
                             shls_slice):
    """Partial shell blocks, only available with ``s4``. When the bra and
    the ket pair select different shell ranges, the ket-side derivative is
    a second integral rather than the transpose of the bra-side one; the
    last block instead has all four indices on one atom, so its derivative
    vanishes.
    """
    mol0 = pyscf.M(atom=atom, basis=basis, unit=unit)
    eri = int2e_packed(coords, basis, shls_slice=shls_slice)
    assert abs(eri - mol0.intor("int2e", aosym="s4",
                                shls_slice=shls_slice)).max() < 1e-10

    # the sliced derivative is the matching block of the full one
    i0, i1, k0, k1 = shls_slice[0], shls_slice[1], shls_slice[4], shls_slice[5]
    ao_loc = mol0.ao_loc_nr()
    jac0 = restore("s4", int2e_jac_ref[ao_loc[i0]:ao_loc[i1],
                                       ao_loc[i0]:ao_loc[i1],
                                       ao_loc[k0]:ao_loc[k1],
                                       ao_loc[k0]:ao_loc[k1]])
    jac = numpy.asarray(jax.jacfwd(int2e_packed)(coords, basis,
                                                 shls_slice=shls_slice))
    assert jac.shape == jac0.shape
    assert abs(jac - jac0).max() < 1e-10

@pytest.mark.parametrize("aosym", ["s4", "s8"])
def test_int2e_nuc_hess(coords, basis, int2e_hess_ref, aosym):
    """Second coordinate derivative. A differentiated integral has lost the
    permutation symmetry of ``(ij|kl)``, so every center contributes its own
    derivative integral, unpacked, and the packed elements are selected.
    """
    fn = partial(int2e_packed, aosym=aosym)

    hess0 = restore(aosym, int2e_hess_ref)
    hess = numpy.asarray(jax.jacfwd(jax.jacfwd(fn))(coords, basis))
    assert hess.shape == hess0.shape
    assert abs(hess - hess0).max() < 1e-10

def test_int2e_nuc_hess_shls_slice(coords, basis, atom, unit, int2e_hess_ref):
    """Second derivative of a block whose bra and ket pair span different
    shells: its first derivative needs both ``int2e_dr1000`` and
    ``int2e_dr0010``, whose own derivatives take opposite branches of the
    transposed-term shortcut.
    """
    shls_slice = SLICES_2E[0]
    i0, i1, k0, k1 = shls_slice[0], shls_slice[1], shls_slice[4], shls_slice[5]
    ao_loc = pyscf.M(atom=atom, basis=basis, unit=unit).ao_loc_nr()

    hess0 = restore("s4", int2e_hess_ref[ao_loc[i0]:ao_loc[i1],
                                        ao_loc[i0]:ao_loc[i1],
                                        ao_loc[k0]:ao_loc[k1],
                                        ao_loc[k0]:ao_loc[k1]])
    fn = partial(int2e_packed, shls_slice=shls_slice)
    hess = numpy.asarray(jax.jacfwd(jax.jacfwd(fn))(coords, basis))
    assert hess.shape == hess0.shape
    assert abs(hess - hess0).max() < 1e-10

def test_int2e_nuc_deriv3_high_cost(coords, basis, atom, unit):
    """Third coordinate derivative, where every center of the twice
    differentiated integrals contributes its own derivative integral.
    """
    mol = Mole()
    mol.atom = atom
    mol.basis = basis
    mol.unit = unit
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    d3 = jax.jacfwd(jax.jacfwd(jax.jacfwd(
        lambda m: m.intor("int2e", aosym="s1"))))(mol)
    ref = restore("s8", numpy.asarray(d3.coords.coords.coords))

    fn = partial(int2e_packed, aosym="s8")
    got = numpy.asarray(jax.jacfwd(jax.jacfwd(jax.jacfwd(fn)))(coords, basis))
    assert got.shape == ref.shape
    assert abs(got - ref).max() < 1e-10

@pytest.mark.skipif("int4c1e_dr1000" not in _INTOR_FUNCTIONS,
                    reason="libcint provides no int4c1e derivatives: it is "
                           "built without WITH_4C1E and auto_intor_ad.cl "
                           "generates no int4c1e_dr names")
@pytest.mark.parametrize("aosym", ["s4", "s8"])
def test_int4c1e_nuc_grad(coords, basis, aosym):
    """``int4c1e`` carries the same 8-fold permutation symmetry as ``int2e``
    and goes through the same packed derivative machinery. Only the libcint
    kernels are missing.
    """
    fn = partial(int2e_packed, aosym=aosym, intor="int4c1e")

    tangent = numpy.asarray(rng_tangent(coords.shape))
    jvp = numpy.asarray(jax.jvp(lambda x: fn(x, basis), (coords,),
                                (np.asarray(tangent),))[1])

    def at(disp):
        return numpy.asarray(fn(coords + disp * tangent, basis))
    d = 1e-5
    fd = (8. * (at(d) - at(-d)) - (at(2*d) - at(-2*d))) / (12. * d)
    assert abs(jvp - fd).max() < 1e-8

@pytest.mark.parametrize("aosym", ["s1", "s2ij", "s2kl"])
def test_int2e_unsupported_aosym(coords, basis, aosym):
    with pytest.raises(NotImplementedError):
        jax.grad(lambda x: np.linalg.norm(
            MoleLite(symbols=("h1", "h2"), coords=x,
                     basis=basis).intor("int2e", aosym=aosym)))(coords)
