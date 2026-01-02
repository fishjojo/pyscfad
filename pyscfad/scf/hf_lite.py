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

from __future__ import annotations
from typing import Any
import numpy

import jax
from jax.lax import while_loop, custom_root
from jax import scipy as jsp

from pyscf.scf.hf import (
    SCF as SCFBase,
    TIGHT_GRAD_CONV_TOL,
)

from pyscfad import numpy as np
from pyscfad.gto.mole_lite import Mole
from pyscfad import lib
from pyscfad.lib import logger
from pyscfad.scf import hf
from pyscfad.scf.diis import SCF_DIIS
from pyscfad.scf.anderson import Anderson
from pyscfad.tools.linear_solver import gen_gmres

Array = Any

def update_dm(
    mf: SCF,
    h1e: Array,
    s1e: Array,
    vhf: Array,
    dm: Array,
    cycle: int,
    diis: Any,
    fock: Array,
    e_tot: float,
) -> tuple[float, float, Array, Array, Array, float, Any]:
    """Single SCF step updating the density matrix.

    Note
    ----
    This function generally has side effects to ``diis``.
    """
    log = logger.new_logger(mf)

    mol = mf.mol
    dm_last = dm
    last_hf_e = e_tot
    fock_last = fock

    mo_energy, mo_coeff = mf.eig(fock_last, s1e)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    vhf = mf.get_veff(mol, dm, dm_last, vhf, s1e=s1e)
    e_tot = mf.energy_tot(dm, h1e, vhf)

    fock = mf.get_fock(h1e, s1e, vhf, dm)
    norm_gorb = np.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
    if not TIGHT_GRAD_CONV_TOL:
        norm_gorb = norm_gorb / np.sqrt(norm_gorb.size)
    de = e_tot-last_hf_e
    log.info("cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g",
             cycle+1, e_tot, de, norm_gorb, np.linalg.norm(dm-dm_last))

    fock = mf.get_fock(h1e, s1e, vhf, dm, cycle+1, diis, fock_last=fock_last)
    del log
    return de, norm_gorb, dm, vhf, fock, e_tot, diis

def _scf(
    mf: SCF,
    dm: Array,
    h1e: Array,
    s1e: Array,
    vhf: Array,
    e_tot: float,
    conv_tol: float,
    conv_tol_grad: float,
) -> tuple[Array, tuple[Array, Array, float]]:
    def cond_fun(value):
        cycle, de, norm_gorb = value[:3]
        return (cycle < mf.max_cycle) & ((abs(de) > conv_tol) | (norm_gorb > conv_tol_grad))

    def body_fun(value):
        cycle, _, _, dm, vhf, fock, e_tot, diis = value
        de, norm_gorb, dm, vhf, fock, e_tot, diis = \
            update_dm(mf, h1e, s1e, vhf, dm, cycle, diis, fock, e_tot)
        return cycle+1, de, norm_gorb, dm, vhf, fock, e_tot, diis

    fock = mf.get_fock(h1e, s1e, vhf, dm)
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
    conv_tol_grad: float,
) -> tuple[Array, tuple[Array, Array, float]]:
    oracle = lambda fn, dm0: _scf(mf, dm0, h1e, s1e, vhf, e_tot, conv_tol, conv_tol_grad)

    def root_fn(dm):
        vhf = mf.get_veff(mf.mol, dm, s1e=s1e)
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm_new = mf.make_rdm1(mo_coeff, mo_occ)
        del mo_energy, mo_occ
        return dm_new - dm

    solver = gen_gmres()
    def tangent_solve(g, dm_bar):
        # FIXME restore the following once issue
        # (https://github.com/jax-ml/jax/issues/33872) is fixed
        #_, vjp_fn = jax.vjp(g, dm_bar)
        #return solver(lambda u: vjp_fn(u)[0], dm_bar)[0]
        assert dm_bar.ndim == 2
        n = dm_bar.shape[-1]
        n2 = n * n
        return jsp.linalg.solve(
            jax.jacobian(g)(dm_bar).reshape((n2,n2)),
            dm_bar.ravel(),
        ).reshape(dm_bar.shape)

    dm_cnvg, (vhf_cnvg, fock_cnvg, e_tot_cnvg) = \
        custom_root(root_fn, dm, oracle, tangent_solve, has_aux=True)
    return dm_cnvg, vhf_cnvg, fock_cnvg, e_tot_cnvg

def kernel(
    mf: SCF,
    conv_tol: float = 1e-10,
    conv_tol_grad: float = None,
    dm0: Array | None = None,
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

    h1e = mf.get_hcore(mol, s1e=s1e)
    vhf = mf.get_veff(mol, dm, s1e=s1e)
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

    dm, _, _, e_tot = mf.scf_implicit(
        dm,
        h1e,
        s1e,
        vhf,
        e_tot,
        conv_tol,
        conv_tol_grad,
    )

    vhf = mf.get_veff(mol, dm, s1e=s1e)
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

class SCF(SCFBase):
    DIIS = SCF_DIIS
    use_sp2 = False
    conv_tol_dm = None
    scf_implicit = _scf_implicit

    def __init__(
        self,
        mol: Mole,
        **kwargs,
    ):
        self.mol = mol
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None
        self.converged = False
        self.verbose = mol.verbose

        self._eri = None
        self.scf_summary = {}

    def dump_flags(self, verbose=None):
        pass

    def build(self, mol=None):
        pass

    def scf(self, dm0=None, **kwargs):
        self.dump_flags()
        self.build(self.mol)
        if self.max_cycle > 0 or self.mo_coeff is None:
            if self.use_sp2:
                from pyscfad.scf import sp2
                self.converged, self.e_tot, \
                        self.mo_energy, self.mo_coeff, self.mo_occ = \
                        sp2.scf(self, conv_tol=self.conv_tol, conv_tol_dm=self.conv_tol_dm,
                                dm0=dm0, **kwargs)
            else:
                self.converged, self.e_tot, \
                        self.mo_energy, self.mo_coeff, self.mo_occ = \
                        kernel(self, conv_tol=self.conv_tol, conv_tol_grad=self.conv_tol_grad,
                               dm0=dm0, **kwargs)
        else:
            self.e_tot = kernel(self, conv_tol=self.conv_tol, conv_tol_grad=self.conv_tol_grad,
                                dm0=dm0, **kwargs)[1]
        return self.e_tot

    def kernel(self, dm0=None, **kwargs):
        return self.scf(dm0=dm0, **kwargs)

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
             fock_last=None):
        if h1e is None:
            h1e = self.get_hcore()
        if vhf is None:
            vhf = self.get_veff(self.mol, dm, s1e=s1e)

        # hack for DFT
        vhf = getattr(vhf, "vxc", vhf)
        f = h1e + vhf
        if diis is None:
            pass

        elif isinstance(diis, Anderson):
            f_tril = diis.update(lib.pack_tril(f), lib.pack_tril(fock_last))
            f = lib.unpack_tril(f_tril)

        else:
            raise NotImplementedError(f"Unsupported diis type {type(diis)}")

        return f

    def get_hcore(self, mol=None, **kwargs):
        return super().get_hcore(mol)

    get_init_guess = hf.SCF.get_init_guess
    get_jk = hf.SCF.get_jk
    get_veff = hf.SCF.get_veff
    make_rdm1 = hf.SCF.make_rdm1
    energy_elec = hf.SCF.energy_elec
    energy_nuc = hf.SCF.energy_nuc
    _eigh = hf.SCF._eigh
