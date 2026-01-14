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

from __future__ import annotations
from typing import Any
from functools import partial

import numpy
from jax.lax import cond, while_loop, custom_root

from pyscfad import numpy as np
from pyscfad import lib
from pyscfad.lib import logger
from pyscfad.scf.anderson import Anderson
#from pyscfad.tools.linear_solver import gen_gmres
from pyscfad.scipy.sparse.linalg import gmres_const_atol

from pyscfad.scf.hf import SCF
Array = Any

def gershgorin_real_bounds(a):
    a = np.asarray(a)
    centers = a.diagonal()
    radii = np.sum(np.abs(a), axis=1) - np.abs(centers)
    lower_bounds = centers.real - radii
    upper_bounds = centers.real + radii
    return np.min(lower_bounds), np.max(upper_bounds)

def update_sp2_dm(dm, dm2, traceP, tracePP, nelectron):
    def _lower(dm, dm2):
        return dm2

    def _raise(dm, dm2):
        return 2 * dm - dm2

    dm_new = cond(
        np.greater(traceP, nelectron),
        _lower,
        _raise,
        dm, dm2
    )
    return dm_new

#def dot2(x):
#    x0 = x.astype(np.float32)
#    x1 = (x - x0).astype(np.float32)
#    a = np.dot(x0, x0).astype(np.float64)
#    b = np.dot(x0, x1).astype(np.float64)
#    #c = np.dot(x1, x1).astype(np.float32)
#    return a + b + b.T

#def dot2(a):
#    a_diag = np.diag(a)
#    a_offdiag = (a - np.diag(a_diag)).astype(np.float32)
#    a_offdiag2 = np.dot(a_offdiag, a_offdiag)
#    a_diag2 = a_diag * a_diag
#    a_diag_single = a_diag.astype(np.float32)
#    tmp = a_diag_single[:,None] * a_offdiag2 + a_offdiag2 * a_diag_single[None,:]
#    return np.diag(a_diag2) + tmp + a_offdiag2

def dot2(a):
    return np.dot(a, a)

def _sp2(dm, nelectron, max_cycle=50, tol=1e-7):
    def cond_fun(value):
        cycle, _, _, traceP, tracePP = value
        return (cycle < max_cycle) & (np.abs(tracePP-traceP) > tol)

    def body_fun(value):
        cycle, dm, dm2, traceP, tracePP = value
        dm_new = update_sp2_dm(dm, dm2, traceP, tracePP, nelectron)
        dm2_new = dot2(dm_new)
        traceP_new = np.trace(dm_new)
        tracePP_new = np.trace(dm2_new)
        return cycle+1, dm_new, dm2_new, traceP_new, tracePP_new

    dm2 = dot2(dm)
    traceP = np.trace(dm)
    tracePP = np.trace(dm2)
    init_val = (0, dm, dm2, traceP, tracePP)
    dm = while_loop(cond_fun, body_fun, init_val)[1]
    return dm

def sp2(h, nelectron, max_cycle=50, tol=1e-7):
    n = h.shape[-1]
    theta = nelectron / n
    theta_bar = 1 - theta

    mu = np.trace(h) / n
    e_min, e_max = gershgorin_real_bounds(h)

    beta = theta / (e_max - mu)
    beta_bar = theta_bar / (mu - e_min)

    beta1 = theta
    beta2 = np.minimum(beta, beta_bar)

    dm0 = -beta2 * h
    dm0 += (beta1 + beta2 * mu) * np.eye(n)

    dm = _sp2(dm0, nelectron, max_cycle=max_cycle, tol=tol)
    return dm

def _ao2mo(a, x):
    return np.dot(x.T.conj(), np.dot(a, x))

def _mo2ao(a, x):
    return np.dot(x, np.dot(a, x.T.conj()))

def update_dm(
    mf: SCF,
    h1e: Array,
    s1e: Array,
    vhf: Array,
    dm: Array,
    cycle: int,
    diis: Any,
    fock: Array,
    x: Array,
    e_tot: float,
) -> tuple[float, float, Array, Array, Array, float, Any]:
    log = logger.new_logger(mf)

    mol = mf.mol
    dm_last = dm
    last_hf_e = e_tot
    fock_last = fock

    h = _ao2mo(fock_last, x)
    nocc = mf.tot_electrons // 2 #FIXME
    dm_orth = sp2(h, nocc)
    dm = _mo2ao(dm_orth, x) * 2

    vhf = mf.get_veff(mol, dm, dm_last, vhf, s1e=s1e)
    e_tot = mf.energy_tot(dm, h1e, vhf)

    de = e_tot-last_hf_e
    norm_ddm = np.linalg.norm(dm-dm_last)
    log.info("cycle= %d E= %.15g  delta_E= %4.3g  |ddm|= %4.3g",
             cycle+1, e_tot, de, norm_ddm)

    fock = mf.get_fock(h1e, s1e, vhf, dm, cycle+1, diis, fock_last=fock_last)
    del log
    return de, norm_ddm, dm, vhf, fock, e_tot, diis

def _scf(
    mf: SCF,
    dm: Array,
    h1e: Array,
    s1e: Array,
    vhf: Array,
    e_tot: float,
    conv_tol: float,
    conv_tol_dm: float,
):
    fock = mf.get_fock(h1e, s1e, vhf, dm)
    _, x = mf.eig(fock, s1e)

    def cond_fun(value):
        cycle, de, norm_ddm = value[:3]
        return (cycle < mf.max_cycle) & ((abs(de) > conv_tol) | (norm_ddm > conv_tol_dm))

    def body_fun(value):
        cycle, _, _, dm, vhf, fock, e_tot, diis = value
        de, norm_ddm, dm, vhf, fock, e_tot, diis = \
            update_dm(mf, h1e, s1e, vhf, dm, cycle, diis, fock, x, e_tot)
        return cycle+1, de, norm_ddm, dm, vhf, fock, e_tot, diis

    if isinstance(mf.diis, str) and mf.diis.lower() == "anderson":
        diis = Anderson(
            lib.pack_tril(fock),
            space=mf.diis_space,
            damp=mf.diis_damp,
            start_cycle=mf.diis_start_cycle
        )
    else:
        diis = None
    init_val = (0, e_tot, 1e3, dm, vhf, fock, e_tot, diis)
    cycle, _, _, dm, vhf, fock, e_tot, _ = while_loop(cond_fun, body_fun, init_val)
    return dm, (vhf, fock, e_tot)

def _scf_implicit(
    mf: SCF,
    dm: Array,
    h1e: Array,
    s1e: Array,
    vhf: Array,
    e_tot: float,
    conv_tol: float,
    conv_tol_dm: float,
):
    oracle = lambda fn, dm0: _scf(mf, dm0, h1e, s1e, vhf, e_tot, conv_tol, conv_tol_dm)

    def root_fn(dm):
        vhf = mf.get_veff(mf.mol, dm, s1e=s1e)
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm_new = mf.make_rdm1(mo_coeff, mo_occ)
        del mo_energy, mo_occ
        return dm_new - dm

    #solver = gen_gmres()
    solver = partial(gmres_const_atol,
                 tol=1e-6, atol=1e-6, maxiter=30,
                 solve_method="batched", restart=20)
    def tangent_solve(g, dm_bar):
        return solver(g, dm_bar)[0]

    dm_cnvg, (vhf_cnvg, fock_cnvg, e_tot_cnvg) = \
        custom_root(root_fn, dm, oracle, tangent_solve, has_aux=True)
    return dm_cnvg, vhf_cnvg, fock_cnvg, e_tot_cnvg

def scf(
    mf: SCF,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = None,
    dm0: Array | None = None,
):
    log = logger.new_logger(mf)
    if conv_tol_dm is None:
        conv_tol_dm = numpy.sqrt(conv_tol)
        log.info("Set dm conv threshold to %g", conv_tol_dm)

    mol = mf.mol
    s1e = mf.get_ovlp(mol)

    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess, s1e=s1e)
    else:
        dm = dm0

    h1e = mf.get_hcore(mol, s1e=s1e)
    vhf = mf.get_veff(mol, dm, s1e=s1e)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    log.info("init E= %.15g", e_tot)

    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None

    dm, vhf, fock, e_tot = _scf_implicit(
        mf,
        dm,
        h1e,
        s1e,
        vhf,
        e_tot,
        conv_tol,
        conv_tol_dm,
    )

    mo_energy, mo_coeff = mf.eig(fock, s1e)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
    vhf = mf.get_veff(mol, dm, dm_last, vhf, s1e=s1e)
    e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

    norm_ddm = np.linalg.norm(dm - dm_last)
    conv_tol = conv_tol * 10
    conv_tol_dm = conv_tol_dm * 3
    scf_conv = (abs(e_tot-last_hf_e) < conv_tol) & (norm_ddm < conv_tol_dm)
    log.info("Extra cycle  E= %.15g  delta_E= %4.3g  |ddm|= %4.3g",
             e_tot, e_tot-last_hf_e, norm_ddm)

    del log
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ
