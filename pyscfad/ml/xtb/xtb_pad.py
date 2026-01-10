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

"""
Molecular XTB with padding
"""
from __future__ import annotations
from pyscfad.scf import hf_lite
from pyscfad.tools.linear_solver import gen_gmres
from pyscfad.ml.xtb.param import ParamArray
from pyscfad.ml.scf import SCFPad
from pyscfad.ml.gto import MolePad
from pyscfad.xtb.data.elements import N_VALENCE_ARRAY
from pyscfad.xtb import util
from pyscfad.xtb.xtb import GFN1XTB as GFN1XTBBase
from pyscfad.xtb.xtb import XTB as XTBBase
from pyscfad.lib import logger
from pyscfad import numpy as np
from pyscf.scf.hf import TIGHT_GRAD_CONV_TOL
from pyscf.gto.mole import ANG_OF
from jax import scipy as jsp
from jax.lax import while_loop, custom_root
import jax
from typing import Any
import numpy
Array = Any
<<<<<<< HEAD

=======
>>>>>>> fishjojo-mlxtb


def tot_valence_electrons(mol: MolePad, charge: int = None, nkpts: int = 1):
    if charge is None:
        charge = mol.charge

    nelecs = N_VALENCE_ARRAY[mol.numbers]
    n = np.sum(nelecs) * nkpts - charge
    return n


def dip_moment(mol, dm, unit="Debye", verbose=logger.NOTE):
    from pyscf.data import nist
    log = logger.new_logger(mol, verbose)

    ao_dip = mol.intor_symmetric("int1e_r", comp=3)
    el_dip = np.einsum("xij,ji->x", ao_dip, dm)

    charges = N_VALENCE_ARRAY[mol.numbers]
    coords = np.asarray(mol.atom_coords())
    nucl_dip = np.einsum("i,ix->x", charges.astype(coords.dtype), coords)
    mol_dip = nucl_dip - el_dip

    if unit.upper() == "DEBYE":
        mol_dip *= nist.AU2DEBYE
        log.note("Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f", *mol_dip)
    else:
        log.note("Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f", *mol_dip)
    del log
    return mol_dip

def normalize_tot_charge(mf, q):
    tot = mf.mol.charge
    shl_mask = mf.mol.shl_mask
    nbas = mf.mol.nbas
    return q.at[:nbas].set(
        np.where(shl_mask, q[:nbas] -
                 (np.sum(q[:nbas]) - tot) / np.sum(shl_mask), 0.)
    )

def _scf_q_broyden(
    mf: XTB,
    q: Array,
    dm: Array,
    h1e: Array,
    s1e: Array,
    vhf: Array,
    e_tot: float,
    conv_tol: float,
    conv_tol_grad: float,
) -> tuple[Array, tuple[Array, Array, float]]:
    log = logger.new_logger(mf)

    def safe_inv_norm(v):
        safe_v = np.where(np.abs(v) < 1e-8, 0., v)
        norm_v = np.sqrt(np.sum(safe_v**2))
        return 1 / (1e-12 + norm_v)

    def cond_fun(value):
        cycle, de, norm_gorb = value[:3]
        return (cycle < mf.max_cycle) & ((abs(de) > conv_tol) | (norm_gorb > conv_tol_grad))

    def body_fun(value):
        cycle, _, _, q1, dq0, dm_last, vhf, fock, last_hf_e, u_hist, v_hist = value

        # --- new dm from last fock ---
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        # FIXME possible to avoid fock build when computing energy?
        vhf = mf.get_veff(dm=dm, s1e=s1e)
        e_tot = mf.energy_tot(dm=dm, h1e=h1e, vhf=vhf)
        # get q from dm
        q = mf.get_q(mol=mf.mol, dm=dm, s1e=s1e)

        # --- broyden step ---
        # ref: https://math.leidenuniv.nl/reports/files/2003-06.pdf
        g1 = q - q1

        inv_norm = safe_inv_norm(dq0)
        u_hist = u_hist.at[:, cycle-1].set(g1 * inv_norm)
        v_hist = v_hist.at[:, cycle-1].set(dq0 * inv_norm)

        A = np.eye(mf.max_cycle) - np.dot(v_hist.T, u_hist)
        b = np.dot(v_hist.T, g1)
        x = np.linalg.solve(A, b)

        dq1 = g1 + np.dot(u_hist, x)
        q2 = normalize_tot_charge(
            mf,
            mf.diis_damp * q1 + (1 - mf.diis_damp) * (q1 + dq1))

        # --- new fock from new q ---
        vhf = mf.get_veff_fromq(q2, mol=mf.mol, s1e=s1e)

        fock = h1e + vhf.vxc
        norm_gorb = np.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        de = e_tot - last_hf_e
        log.info("cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g",
                 cycle+1, e_tot, de, norm_gorb, np.linalg.norm(dm-dm_last))

        return cycle+1, de, norm_gorb, q2, dq1, dm, vhf, fock, e_tot, u_hist, v_hist

    # first cycle only does damping
    dm_last = dm
    last_hf_e = e_tot
    fock = h1e + vhf.vxc
    mo_energy, mo_coeff = mf.eig(fock, s1e)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    # FIXME possible to avoid fock build when computing energy?
    vhf = mf.get_veff(dm=dm, s1e=s1e)
    e_tot = mf.energy_tot(dm=dm, h1e=h1e, vhf=vhf)
    q1 = mf.get_q(mol=mf.mol, dm=dm, s1e=s1e)
    dq0 = q1 - q
    q1 = normalize_tot_charge(mf, mf.diis_damp * q + (1 - mf.diis_damp) * q1)
    vhf = mf.get_veff_fromq(q1, mol=mf.mol, s1e=s1e)
    fock = h1e + vhf.vxc
    norm_gorb = np.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
    de = e_tot - last_hf_e
    log.info("cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g",
             1, e_tot, de, norm_gorb, np.linalg.norm(dm-dm_last))

    n_dim = q.size
    u_hist = np.zeros((n_dim, mf.max_cycle))
    v_hist = np.zeros((n_dim, mf.max_cycle))

    init_val = (1, de, norm_gorb, q1, dq0, dm, vhf, fock, e_tot, u_hist, v_hist)
    cycle, _, _, q, dq, dm, vhf, fock, e_tot, _, _ = while_loop(
        cond_fun, body_fun, init_val)

    del log
    return q, (dm, dq, fock, e_tot)


def _scf_implicit_q(
    mf: XTB,
    q: Array,
    dm: Array,
    h1e: Array,
    s1e: Array,
    vhf: Array,
    e_tot: float,
    conv_tol: float,
    conv_tol_grad: float,
) -> tuple[Array, tuple[Array, Array, float]]:
    if mf.diis.lower() == 'broyden':
        def oracle(fn, q0): return _scf_q_broyden(
            mf, q0, dm, h1e, s1e, vhf, e_tot, conv_tol, conv_tol_grad)
    else:
        raise NotImplementedError

    def root_fn(q):
        vhf = mf.get_veff_fromq(q, mf.mol, s1e=s1e)
        fock = h1e + vhf.vxc
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm_new = mf.make_rdm1(mo_coeff, mo_occ)
        q_new = mf.get_q(mol=mf.mol, dm=dm_new, s1e=s1e)
        del mo_energy, mo_occ
        return q_new - q

    def tangent_solve(g, q_bar):
        return jsp.linalg.solve(
            jax.jacobian(g)(q_bar),
            q_bar,
        )

    q_cnvg, (dm, dq, fock, e_tot) \
        = custom_root(root_fn, normalize_tot_charge(mf, q), oracle, tangent_solve, has_aux=True)

    return q_cnvg, dm, fock, e_tot

def kernel(
    mf: SCF,
    conv_tol: float = 1e-10,
    conv_tol_grad: float = None,
    dm0: Array | None = None,
    q0: Array | None = None,
) -> tuple[bool, float, Array, Array, numpy.ndarray]:
    log = logger.new_logger(mf)
    #cput0 = log.get_t0()
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        log.info("Set gradient conv threshold to %g", conv_tol_grad)

    mol = mf.mol
    s1e = mf.get_ovlp(mol)

    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess, s1e=s1e)
    else:
        dm = dm0

    if q0 is None:
        q = mf.get_q(mol=mol, dm=dm, s1e=s1e)
    else:
        q = q0

    h1e = mf.get_hcore(mol, s1e=s1e)
    vhf = mf.get_veff_fromq(q, mol=mol, s1e=s1e)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    log.info("init E= %.15g", e_tot)

    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None

    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        # hack for ROHF
        mo_energy = getattr(mo_energy, "mo_energy", mo_energy)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    #cput1 = log.timer('initialize scf', *cput0)

    q, dm, _, e_tot = _scf_implicit_q(
        mf,
        q,
        dm,
        h1e,
        s1e,
        vhf,
        e_tot,
        conv_tol,
        conv_tol_grad,
    )

    vhf = mf.get_veff_fromq(q, mol=mol, s1e=s1e)
    fock = mf.get_fock(h1e, s1e, vhf, dm)
    mo_energy, mo_coeff = mf.eig(fock, s1e)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
    vhf = mf.get_veff(mol, dm, dm_last, vhf, s1e=s1e)
    e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

    fock = mf.get_fock(h1e, s1e, vhf, dm)
    norm_gorb = np.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
    if not TIGHT_GRAD_CONV_TOL:
        norm_gorb = norm_gorb / np.sqrt(norm_gorb.size)
    norm_ddm = np.linalg.norm(dm - dm_last)

    conv_tol = conv_tol * 10
    conv_tol_grad = conv_tol_grad * 3
    scf_conv = (abs(e_tot-last_hf_e) < conv_tol) & (norm_gorb < conv_tol_grad)
    log.info("Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g",
             e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)

    #log.timer('scf_cycle', *cput0)
    del log
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


class XTB(XTBBase, SCFPad):
    @property
    def tot_electrons(self):
        return tot_valence_electrons(self.mol)

    def dip_moment(self, mol=None, dm=None, unit="Debye", verbose=None,
                   **kwargs):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if verbose is None:
            verbose = mol.verbose
        return dip_moment(mol, dm, unit, verbose=verbose)

    get_occ = SCFPad.get_occ


class XTBQ(XTB):
    diis = 'broyden'
    diis_damp = 0.6
    scf_implicit = _scf_implicit_q

    def scf(self, dm0=None, q0=None, **kwargs):
        self.dump_flags()
        self.build(self.mol)
        if self.max_cycle > 0 or self.mo_coeff is None:
            self.converged, self.e_tot, \
                    self.mo_energy, self.mo_coeff, self.mo_occ = \
                    kernel(self, conv_tol=self.conv_tol, conv_tol_grad=self.conv_tol_grad,
                           dm0=dm0, q0=q0, **kwargs)
        else:
            self.e_tot = kernel(self, conv_tol=self.conv_tol, conv_tol_grad=self.conv_tol_grad,
                                dm0=dm0, q0=q0, **kwargs)[1]
        return self.e_tot

    def kernel(self, dm0=None, q0=None, **kwargs):
        return self.scf(dm0=dm0, q0=q0, **kwargs)

class GFN1XTB(GFN1XTBBase, XTB):
    def _get_gamma(self):
        gamma = GFN1XTBBase._get_gamma(self)

        shl_mask = self.mol.shl_mask
        mask = np.outer(shl_mask, shl_mask)
        gamma = np.where(mask, gamma, 0.)
        return gamma

    get_occ = XTB.get_occ
    dip_moment = XTB.dip_moment


class GFN1XTBQ(XTBQ, GFN1XTB):
    pass


if __name__ == "__main__":
    import jax
    from pyscfad.xtb import basis as xtb_basis
    from pyscfad.ml.gto import make_basis_array
    from pyscfad.ml.xtb.param import make_param_array

    bfile = xtb_basis.get_basis_filename()
    basis = make_basis_array(bfile, max_number=8)
    param = make_param_array(basis, max_number=8)

    numbers = np.array(
        [
            [8, 1, 1, 0, 0],
            [7, 1, 1, 1, 0],
        ],
        dtype=np.int32
    )
    coords = np.array(
        [
            np.array([
                [0.00000,  0.00000,  0.00000],
                [1.43355,  0.00000, -0.95296],
                [1.43355,  0.00000,  0.95296],
                [0.00000,  0.00000,  0.00000],
                [1.00000,  0.00000,  0.00000],
            ]),
            np.array([
                [-0.80650, -1.00659,  0.02850],
                [-0.50540, -0.31299,  0.68220],
                [0.00620, -1.41579, -0.38500],
                [-1.32340, -0.54779, -0.69350],
                [0.00000,  0.00000,  0.00000],
            ]) / 0.52917721067121,
        ]
    )

    def energy(numbers, coords):
        mol = MolePad(numbers, coords, basis=basis,
                      verbose=4, trace_coords=True)
        mf = GFN1XTB(mol, param)
        mf.diis = "anderson"
        mf.conv_tol = 1e-10
        mf.diis_damp = 0.5
        mf.diis_space = 6
        # mf.sigma = 0.001
        e = mf.kernel()
        mu = mf.dip_moment()
        r2 = mol.intor("int1e_r2", hermi=1)
        e_r2 = np.einsum("ij,ij->", mf.make_rdm1(), r2)
        e_homo, e_lumo = mf.get_homo_lumo_energy()
        return e, {"dip": mu, "r2": e_r2, "e_homo": e_homo, "e_lumo": e_lumo}

    gfn = jax.value_and_grad(energy, 1, has_aux=True)
    (e, aux_res), g = jax.jit(jax.vmap(gfn))(numbers, coords)
    print(e)
    print(g)
    print(aux_res)

    def energy_broyden(numbers, coords):
        mol = MolePad(numbers, coords, basis=basis,
                      verbose=4, trace_coords=True)
        mf = GFN1XTBQ(mol, param)
        mf.diis = "broyden"
        mf.conv_tol = 1e-10
        mf.diis_damp = 0.6
        # mf.sigma = 0.001
        e = mf.kernel()
        mu = mf.dip_moment()
        r2 = mol.intor("int1e_r2", hermi=1)
        e_r2 = np.einsum("ij,ij->", mf.make_rdm1(), r2)
        e_homo, e_lumo = mf.get_homo_lumo_energy()
        return e, {"dip": mu, "r2": e_r2, "e_homo": e_homo, "e_lumo": e_lumo}

    (eb, auxb), gb = jax.jit(jax.vmap(jax.value_and_grad(
        energy_broyden, 1, has_aux=True)))(numbers, coords)
    assert np.abs(e - eb).max() < 1e-8
    assert np.abs(g - gb).max() < 1e-6
    for k in aux_res.keys():
        assert np.abs(aux_res[k] - auxb[k]).max() < 1e-4
