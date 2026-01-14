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

import jax
from jax.lax import while_loop, custom_root
#from jax import scipy as jsp

from pyscf.data import nist
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
#from pyscfad.tools.linear_solver import gen_gmres
from pyscfad.scipy.sparse.linalg import gmres_const_atol
from pyscfad.scf import addons

Array = Any

def get_occ(
    mf: SCF,
    mo_energy: Array | None = None,
    mo_coeff: Array | None = None,
) -> Array:
    """Get MO occupations.
    """
    # NOTE assuming mo_energy is in ascending order
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    nmo = mo_energy.size

    mask = mf.mo_mask(mo_energy, mo_coeff)
    nocc = mf.tot_electrons // 2

    if mf.sigma is not None and mf.sigma > 0:
        mo_occ = addons.get_occ_smearing(mo_energy, nocc, mf.sigma, mask, method=mf.smearing_method)
        mo_occ *= 2
    else:
        pick = (np.cumsum(mask) <= nocc) & mask
        mo_occ = np.where(pick, 2., 0.)
        if mf.verbose >= logger.DEBUG:
            e_homo = np.max(np.where(pick, mo_energy, -np.inf))
            e_lumo = np.min(np.where(mask & ~pick, mo_energy, np.inf))
            logger.debug(mf, "  HOMO = %.15g  LUMO = %.15g", e_homo, e_lumo)

    if mf.verbose >= logger.DEBUG:
        numpy.set_printoptions(threshold=nmo)
        logger.debug(mf, "  mo_energy =\n%s", mo_energy)
        numpy.set_printoptions(threshold=1000)
    return mo_occ

def get_homo_lumo_energy(
    mf: SCF,
    mo_energy: Array | None = None,
    mo_coeff: Array | None = None,
) -> tuple[float, float]:
    """Get HOMO and LUMO energies.
    """
    if mf.sigma is not None and mf.sigma > 0:
        raise NotImplementedError

    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff

    mask = mf.mo_mask(mo_energy, mo_coeff)
    nocc = mf.tot_electrons // 2
    pick = (np.cumsum(mask) <= nocc) & mask

    e_homo = np.max(np.where(pick, mo_energy, -np.inf))
    e_lumo = np.min(np.where(mask & ~pick, mo_energy, np.inf))
    if mf.verbose >= logger.DEBUG:
        logger.debug(mf, "  HOMO = %.15g  LUMO = %.15g", e_homo, e_lumo)
    return e_homo, e_lumo

def get_grad(mo_coeff, mo_occ, fock_ao):
    fock_mo = mo_coeff.conj().T @ fock_ao @ mo_coeff
    occ_mask = np.where(mo_occ > 0, 1, 0)
    vir_mask = 1 - occ_mask
    g = 2 * fock_mo * (vir_mask[:,None] * occ_mask[None,:])
    return g.ravel()

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

    Notes
    -----
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
        return dm_new - dm

    # FIXME restore to use jax gmres once issue
    # (https://github.com/jax-ml/jax/issues/33872) is fixed
    solver = partial(gmres_const_atol,
                     tol=1e-6, atol=1e-6, maxiter=30,
                     solve_method="batched", restart=20)
    def tangent_solve(g, dm_bar):
        return solver(g, dm_bar)[0]
        #assert dm_bar.ndim == 2
        #n = dm_bar.shape[-1]
        #n2 = n * n
        #return jsp.linalg.solve(
        #    jax.jacobian(g)(dm_bar).reshape((n2,n2)),
        #    dm_bar.ravel(),
        #).reshape(dm_bar.shape)

    dm_cnvg, (vhf_cnvg, fock_cnvg, e_tot_cnvg) = \
        custom_root(root_fn, dm, oracle, tangent_solve, has_aux=True)
    return dm_cnvg, vhf_cnvg, fock_cnvg, e_tot_cnvg

def kernel(
    mf: SCF,
    conv_tol: float = 1e-10,
    conv_tol_grad: float = None,
    dm0: Array | None = None,
) -> tuple[bool, float, Array, Array, Array]:
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

    dm, _, _, e_tot = _scf_implicit(
        mf,
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
    diss = None
    use_sp2 = False
    conv_tol_dm = None
    sigma = None
    smearing_method = "fermi"

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

    @property
    def tot_electrons(self):
        return self.mol.tot_electrons()

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

    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ

        dm = (mo_coeff * mo_occ) @ mo_coeff.conj().T
        return dm

    def mo_mask(
        self,
        mo_energy: Array | None = None,
        mo_coeff: Array | None = None,
    ) -> Array:
        """MO masks.

        Useful for e.g. padding, where there are fake MOs.

        Returns
        -------
        mask : array
            Mask array where elements with ``True`` values
            indicate real MOs.
        """
        if mo_energy is None:
            mo_energy = self.mo_energy
        return np.ones(mo_energy.size, dtype=bool)

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        return get_grad(mo_coeff, mo_occ, fock)

    def dip_moment(
        self,
        mol: Mole | None = None,
        dm: Array | None = None,
        unit: str = "Debye",
        origin: Array | None = None,
        verbose: int | None = None,
        charges: Array | None = None,
    ) -> Array:
        """Molecular dipole moment.

        Parameters
        ----------
        charges : array, optional
            Nuclear charges.
        """
        if mol is None:
            mol = self.mol
        if dm is None:
            dm =self.make_rdm1()
        if verbose is None:
            verbose = mol.verbose

        log = logger.new_logger(mol, verbose)

        if charges is None:
            charges = np.asarray(mol.atom_charges())
        coords  = np.asarray(mol.atom_coords())

        if origin is None:
            origin = np.zeros(3)
        else:
            origin = np.asarray(origin, dtype=float)

        with mol.with_common_orig(origin):
            ao_dip = mol.intor_symmetric("int1e_r", comp=3)
        el_dip = np.einsum("xij,ji->x", ao_dip, dm)

        nucl_dip = np.einsum("i,ix->x", charges.astype(coords.dtype), coords)
        mol_dip = nucl_dip - el_dip

        if unit.upper() == "DEBYE":
            mol_dip *= nist.AU2DEBYE
            log.note("Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f", *mol_dip)
        else:
            log.note("Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f", *mol_dip)
        del log
        return mol_dip

    get_init_guess = hf.SCF.get_init_guess
    get_jk = hf.SCF.get_jk
    get_veff = hf.SCF.get_veff
    energy_elec = hf.SCF.energy_elec
    energy_nuc = hf.SCF.energy_nuc
    _eigh = hf.SCF._eigh
    get_occ = get_occ
    get_homo_lumo_energy = get_homo_lumo_energy
