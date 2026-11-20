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
import pytest
import numpy
import jax
from pyscf import dft as pyscf_dft
from pyscfad import config_update
from pyscfad import dft
from pyscfad.dft import numint, gen_grid, rks_lite, uks_lite, roks_lite
from pyscfad.gto import MoleLite
from .util import (
    hf_energy,
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

def _count_second_call_compilations(fn, *args):
    # Number of XLA compilations triggered by a *repeated* identical call to fn.
    # The first call compiles everything; a well-behaved function reuses those
    # compilations on the second call.
    import logging
    fn(*args)  # warm up: compile everything once
    n = [0]
    class _Counter(logging.Handler):
        def emit(self, record):
            if "Finished XLA compilation of" in record.getMessage():
                n[0] += 1
    # A single handler on the root "jax" logger catches every compile record
    # via logging propagation (jax logs compilations under jax._src.*).
    jax_logger = logging.getLogger("jax")
    handler = _Counter()
    prev_flag = jax.config.jax_log_compiles
    prev_level = jax_logger.level
    jax.config.update("jax_log_compiles", True)
    jax_logger.setLevel(logging.WARNING)
    jax_logger.addHandler(handler)
    try:
        fn(*args)
    finally:
        jax_logger.removeHandler(handler)
        jax_logger.setLevel(prev_level)
        jax.config.update("jax_log_compiles", prev_flag)
    return n[0]

def test_rks_repeated_grad_no_recompile(mol_H2):
    # Regression test for issue #99: repeated jax.grad / jax.value_and_grad with
    # dft.RKS used to recompile the AO-on-grid derivative kernels on every call
    # (they were `jit`-decorated closures rebuilt inside the eval_gto JVP rules,
    # and JAX keys its compilation cache on the function object), causing severe
    # wall-time blowup. scf.RHF -- which never evaluates AOs on a grid -- was
    # unaffected and is used here as the "cache-friendly" baseline: after the fix
    # a repeated RKS gradient must trigger no more compilations than RHF.
    from pyscfad import scf
    mol = mol_H2()
    with config_update('pyscfad_scf_implicit_diff', True):
        rhf = _count_second_call_compilations(
            lambda m: jax.grad(hf_energy)(m, scf.RHF), mol)
        rks = _count_second_call_compilations(
            lambda m: jax.grad(dft_energy)(m, dft.RKS, "pbe,pbe"), mol)
    assert rks <= rhf + 4, (
        f"dft.RKS recompiled on a repeated gradient call "
        f"(RKS={rks} vs RHF={rhf} XLA compilations); see issue #99")

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

@pytest.mark.parametrize("xc", LDA + GGA + MGGA)
def test_numint_jit(mol_H2, xc):
    """The XC functional is evaluated through a traceable callback to
    libxc, so the numerical integration can be jitted (issue #175).
    """
    pmol = mol_H2().to_pyscf()
    mol = MoleLite.from_pyscf(pmol)
    grids = pyscf_dft.gen_grid.Grids(pmol).build()
    dm = jax.numpy.asarray(pmol.RHF().get_init_guess())
    ni = numint.NumInt()
    ni0 = pyscf_dft.numint.NumInt()

    n, exc, vxc = jax.jit(lambda dm: ni.nr_rks(mol, grids, xc, dm))(dm)
    n0, exc0, vxc0 = ni0.nr_rks(pmol, grids, xc, numpy.asarray(dm))
    assert abs(n - n0) < 1e-10
    assert abs(exc - exc0) < 1e-10
    assert abs(vxc - vxc0).max() < 1e-10

    # d(exc)/d(dm) through the custom JVP of the XC functional
    g = jax.jit(jax.grad(lambda dm: ni.nr_rks(mol, grids, xc, dm)[1]))(dm)
    assert abs((g + g.T) * .5 - vxc0).max() < 1e-10

    dms = (dm * .6, dm * .4)
    n, exc, vxc = jax.jit(lambda dms: ni.nr_uks(mol, grids, xc, dms))(dms)
    n0, exc0, vxc0 = ni0.nr_uks(pmol, grids, xc, numpy.asarray(dms))
    assert abs(n - n0).max() < 1e-10
    assert abs(exc - exc0) < 1e-10
    assert abs(vxc - vxc0).max() < 1e-10

def test_grids_lite(h2o_lite):
    """The traceable grids reproduce pyscf's (unsorted and unpadded)."""
    symbols, coords = h2o_lite
    mol = MoleLite(symbols, coords, basis="sto3g", verbose=0)
    grids0 = pyscf_dft.gen_grid.Grids(mol.to_pyscf())
    grids0.atom_grid = (20, 50)
    grids0.alignment = 0
    grids0.build(sort_grids=False)

    def build(coords):
        grids = gen_grid.Grids(MoleLite(symbols, coords, basis="sto3g", verbose=0))
        grids.atom_grid = (20, 50)
        grids.build()
        return grids.coords, grids.weights

    coords, weights = jax.jit(build)(coords)
    assert abs(coords - grids0.coords).max() < 1e-12
    assert abs(weights - grids0.weights).max() < 1e-12

KS_LITE = [
    # method, xc, use_sp2
    ("RKS", "pbe,pbe", False),
    ("RKS", "b3lyp", True),
    ("UKS", "pbe,pbe", True),
    ("ROKS", "b3lyp", False),
]

@pytest.mark.parametrize("method,xc,use_sp2", KS_LITE)
def test_ks_lite_nuc_grad(h2o_lite, method, xc, use_sp2):
    """Kohn-Sham DFT with the SCFLite classes and either solver, under jit
    (issue #175). The grids move with the atoms, so the nuclear gradient
    includes the grid response.
    """
    symbols, coords = h2o_lite
    charge = spin = 0 if method == "RKS" else 1
    atom_grid = (30, 110)
    lite = {"RKS": rks_lite.RKS, "UKS": uks_lite.UKS, "ROKS": roks_lite.ROKS}[method]

    def energy(coords):
        mol = MoleLite(symbols, coords, basis="sto3g", charge=charge,
                       spin=spin, verbose=0)
        mf = lite(mol, xc=xc)
        mf.grids.atom_grid = atom_grid
        mf.init_guess = "hcore"
        mf.use_sp2 = use_sp2
        if use_sp2:
            mf.diis = "anderson"
        return mf.kernel()

    pmol = MoleLite(symbols, coords, basis="sto3g", charge=charge,
                    spin=spin, verbose=0).to_pyscf()
    mf0 = getattr(pyscf_dft, method)(pmol, xc=xc)
    mf0.grids.atom_grid = atom_grid
    mf0.small_rho_cutoff = 0
    mf0.conv_tol = 1e-12
    mf0.kernel()
    g0 = mf0.nuc_grad_method()
    g0.grid_response = True

    e, g = jax.jit(jax.value_and_grad(energy))(coords)
    # SP2 stops at |ddm| < sqrt(conv_tol)
    assert abs(e - mf0.e_tot) < (1e-7 if use_sp2 else 1e-9)
    assert abs(g - g0.kernel()).max() < 1e-6

def test_rks_lite_nlc():
    """The VV10 kernel through a traceable callback, whose derivative
    includes the response of the grids.
    """
    coords = numpy.array([[0., 0., 0.], [0., 0., 1.4]])
    xc = "b97m_v"

    def energy(coords):
        mol = MoleLite(("H", "H"), coords, basis="sto3g", verbose=0)
        mf = rks_lite.RKS(mol, xc=xc)
        mf.grids.atom_grid = mf.nlcgrids.atom_grid = (20, 50)
        mf.init_guess = "hcore"
        return mf.kernel()

    pmol = MoleLite(("H", "H"), coords, basis="sto3g", verbose=0).to_pyscf()
    mf0 = pyscf_dft.RKS(pmol, xc=xc)
    mf0.grids.atom_grid = mf0.nlcgrids.atom_grid = (20, 50)
    mf0.small_rho_cutoff = 0
    mf0.conv_tol = 1e-12
    mf0.kernel()
    g0 = mf0.nuc_grad_method()
    g0.grid_response = True

    e, g = jax.jit(jax.value_and_grad(energy))(coords)
    assert abs(e - mf0.e_tot) < 1e-9
    assert abs(g - g0.kernel()).max() < 1e-6

def test_rks_lite_basis_grad():
    """Derivative w.r.t. a basis exponent, through the AO values
    on the grids (issue #175).
    """
    def mol(alpha):
        return MoleLite(("He",), [[0., 0., 0.]], basis={"He": [[0, [alpha, 1.]]]},
                        verbose=0)

    def energy(alpha):
        mf = rks_lite.RKS(mol(alpha), xc="LDA,VWN")
        mf.init_guess = "hcore"
        mf.diis = "anderson"
        return mf.kernel()

    def energy_pyscf(alpha):
        mf = pyscf_dft.RKS(mol(alpha).to_pyscf(), xc="LDA,VWN")
        mf.small_rho_cutoff = 0
        mf.conv_tol = 1e-12
        return mf.kernel()

    disp = 1e-4
    g0 = (energy_pyscf(1. + disp) - energy_pyscf(1. - disp)) / (2 * disp)
    assert abs(jax.jit(jax.grad(energy))(1.) - g0) < 1e-6
