# Copyright 2025-2026 The PySCFAD Authors
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

"""Broyden's method for charge self-consistency
"""
from typing import Any
from functools import partial
import numpy
from jax.lax import while_loop, custom_root
from pyscf.scf.hf import TIGHT_GRAD_CONV_TOL
from pyscfad import numpy as np
from pyscfad.lib import logger
from pyscfad.xtb import XTB
from pyscfad.scipy.sparse.linalg import gmres_const_atol

Array = Any

def safe_inv_norm(v):
    safe_v = np.where(np.abs(v) < 1e-8, 0., v)
    norm_v = np.sqrt(np.sum(safe_v**2))
    return 1 / (1e-12 + norm_v)

def normalize_tot_charge(mol, q):
    tot = mol.charge
    nbas = mol.nbas
    shl_mask = getattr(mol, "shl_mask", np.ones(nbas, dtype=bool))
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
    mol = mf.mol
    diis_damp = mf.diis_damp

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
        q = mf.get_q(mol=mol, dm=dm, s1e=s1e)

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
            mol,
            diis_damp * q1 + (1 - diis_damp) * (q1 + dq1),
        )

        # --- new fock from new q ---
        vhf = mf.get_veff(mol=mol, s1e=s1e, q=q2)

        fock = h1e + vhf.vxc
        norm_gorb = np.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        de = e_tot - last_hf_e
        log.info("cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g",
                 cycle+1, e_tot, de, norm_gorb, np.linalg.norm(dm-dm_last))

        return cycle+1, de, norm_gorb, q2, dq1, dm, vhf, fock, e_tot, u_hist, v_hist

    # first cycle only does damping
    dm_last = dm
    last_hf_e = e_tot
    fock = mf.get_fock(h1e, s1e, vhf, dm)
    mo_energy, mo_coeff = mf.eig(fock, s1e)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    dm = mf.make_rdm1(mo_coeff, mo_occ)

    # FIXME possible to avoid fock build when computing energy?
    vhf = mf.get_veff(dm=dm, s1e=s1e)
    e_tot = mf.energy_tot(dm=dm, h1e=h1e, vhf=vhf)
    q1 = mf.get_q(mol=mol, dm=dm, s1e=s1e)
    dq0 = q1 - q

    q1 = normalize_tot_charge(mol, diis_damp * q + (1 - diis_damp) * q1)
    vhf = mf.get_veff(mol=mol, s1e=s1e, q=q1)
    fock = mf.get_fock(h1e, s1e, vhf, dm)
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
    oracle = lambda fn, q0: _scf_q_broyden(mf, q0, dm, h1e, s1e, vhf, e_tot, conv_tol, conv_tol_grad)

    def root_fn(q):
        vhf = mf.get_veff(mf.mol, s1e=s1e, q=q)
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm_new = mf.make_rdm1(mo_coeff, mo_occ)
        q_new = mf.get_q(mol=mf.mol, dm=dm_new, s1e=s1e)
        return q_new - q

    solver = partial(gmres_const_atol,
                     tol=1e-6, atol=1e-6, maxiter=30,
                     solve_method="batched", restart=20)
    def tangent_solve(g, q_bar):
        return solver(g, q_bar)[0]
        #return jsp.linalg.solve(
        #    jax.jacobian(g)(q_bar),
        #    q_bar,
        #)

    q_cnvg, (dm, dq, fock, e_tot) \
        = custom_root(root_fn, normalize_tot_charge(mf.mol, q), oracle, tangent_solve, has_aux=True)
    return q_cnvg, dm, fock, e_tot


def scf(
    mf: XTB,
    conv_tol: float = 1e-10,
    conv_tol_grad: float = None,
    dm0: Array | None = None,
    q0: Array | None = None,
) -> tuple[bool, float, Array, Array, Array]:
    log = logger.new_logger(mf)
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
    vhf = mf.get_veff(mol=mol, s1e=s1e, q=q)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    log.info("init E= %.15g", e_tot)

    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None

    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

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

    vhf = mf.get_veff(mol=mol, s1e=s1e, q=q)
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

    del log
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ
