# Copyright 2021-2026 The PySCFAD Authors
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
Restricted CCSD.

Two implementations share the amplitude equations of this module
(:func:`amplitude_equation` and :func:`update_amps`):

* :class:`RCCSD` is based on the legacy PySCF-derived
  :class:`~pyscfad.gto.Mole` and
  :class:`~pyscfad.scf.hf.SCF` objects.
* :class:`RCCSDLite`, a fully jittable implementation for the lightweight
  :class:`~pyscfad.gto.MoleLite` / :class:`~pyscfad.ml.gto.MolePad`
  molecules and the :class:`~pyscfad.scf.hf_lite.SCFLite` /
  :class:`~pyscfad.ml.scf.SCFPad` mean fields.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from functools import partial
import sys

from jax.lax import while_loop, custom_root

from pyscf.lib import current_memory

from pyscfad import numpy as np
from pyscfad import ao2mo
from pyscfad import lib
from pyscfad.lib import logger
from pyscfad.lib.diis_lite import DIISLite
from pyscfad.scf.anderson import Anderson
from pyscfad.scipy.sparse.linalg import gmres_const_atol
from pyscfad.cc import ccsd
from pyscfad.cc import rintermediates as imd
from pyscfad.cc.ccsd import energy

if TYPE_CHECKING:
    from typing import Any
    from pyscfad.typing import ArrayLike, Array
    from pyscfad.scf.hf_lite import SCFLite


def mo_energy_denominators(
    mo_e_o: Array,
    mo_e_v: Array,
    mo_mask: Array | None = None,
) -> tuple[Array, Array]:
    r"""Orbital-energy denominators :math:`\Delta_i^a` and
    :math:`\Delta_{ij}^{ab}`.

    Parameters:
        mo_e_o: Occupied orbital energies.
        mo_e_v: Virtual orbital energies.
        mo_mask: Mask flagging the real (non-padding) MOs, as returned by
            :meth:`~pyscfad.scf.hf_lite.SCF.mo_mask`. ``None`` when every MO
            is real.

    Notes:
        A padding MO enters no integral, so its diagonal Fock element, and
        hence its orbital energy, is zero up to round-off. ``eia`` therefore
        vanishes for a pair of padding orbitals -- where the numerator
        vanishes as well -- and the denominator is replaced by one there to
        keep the ratio, and its derivative, finite.
    """
    eia = mo_e_o[:,None] - mo_e_v
    if mo_mask is not None:
        nocc = mo_e_o.shape[0]
        eia = np.where(mo_mask[:nocc,None] | mo_mask[None,nocc:], eia, 1.)
    eijab = eia[:,None,:,None] + eia[None,:,None,:]
    return eia, eijab

def update_amps(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift
    mo_oo = np.diagflat(mo_e_o)
    mo_vv = np.diagflat(mo_e_v)

    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    t1new, t2new = cc.amplitude_equation(t1, t2, eris)
    # Move energy terms to the other side
    t1new +=   np.einsum("ac,ic->ia", -mo_vv, t1)
    t1new +=  -np.einsum("ki,ka->ia", -mo_oo, t1)
    if cc.cc2:
        Lvv2 = -np.diagflat(np.diag(fvv))
        tmp = np.einsum("ac,ijcb->ijab", Lvv2, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        Loo2 = -np.diagflat(np.diag(foo))
        tmp = np.einsum("ki,kjab->ijab", Loo2, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
    else:
        tmp = np.einsum("ac,ijcb->ijab", -mo_vv, t2)
        t2new += tmp + tmp.transpose(1,0,3,2)
        tmp = np.einsum("ki,kjab->ijab", -mo_oo, t2)
        t2new -= tmp + tmp.transpose(1,0,3,2)

    # only the padded ERIs carry an MO mask
    eia, eijab = mo_energy_denominators(mo_e_o, mo_e_v,
                                        getattr(eris, "mo_mask", None))
    t1new /= eia
    t2new /= eijab
    return t1new, t2new

def amplitude_equation(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc,nocc:]
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)

    # T1 equation
    t1new  =-2*np.einsum("kc,ka,ic->ia", fov, t1, t1)
    t1new +=   np.einsum("ac,ic->ia", Fvv, t1)
    t1new +=  -np.einsum("ki,ka->ia", Foo, t1)
    t1new += 2*np.einsum("kc,kica->ia", Fov, t2)
    t1new +=  -np.einsum("kc,ikca->ia", Fov, t2)
    t1new +=   np.einsum("kc,ic,ka->ia", Fov, t1, t1)
    t1new += fov.conj()
    t1new += 2*np.einsum("kcai,kc->ia", eris.ovvo, t1)
    t1new +=  -np.einsum("kiac,kc->ia", eris.oovv, t1)
    eris_ovvv = eris.get_ovvv()
    t1new += 2*np.einsum("kdac,ikcd->ia", eris_ovvv, t2)
    t1new +=  -np.einsum("kcad,ikcd->ia", eris_ovvv, t2)
    t1new += 2*np.einsum("kdac,kd,ic->ia", eris_ovvv, t1, t1)
    t1new +=  -np.einsum("kcad,kd,ic->ia", eris_ovvv, t1, t1)
    eris_ovoo = eris.ovoo
    t1new +=-2*np.einsum("lcki,klac->ia", eris_ovoo, t2)
    t1new +=   np.einsum("kcli,klac->ia", eris_ovoo, t2)
    t1new +=-2*np.einsum("lcki,lc,ka->ia", eris_ovoo, t1, t1)
    t1new +=   np.einsum("kcli,lc,ka->ia", eris_ovoo, t1, t1)

    # T2 equation
    tmp2  = np.einsum("kibc,ka->abic", eris.oovv, -t1)
    tmp2 += np.asarray(eris_ovvv).conj().transpose(1,3,0,2)
    tmp = np.einsum("abic,jc->ijab", tmp2, t1)
    t2new = tmp + tmp.transpose(1,0,3,2)
    tmp2  = np.einsum("kcai,jc->akij", eris.ovvo, t1)
    tmp2 += eris_ovoo.transpose(1,3,0,2).conj()
    tmp = np.einsum("akij,kb->ijab", tmp2, t1)
    t2new -= tmp + tmp.transpose(1,0,3,2)
    t2new += np.asarray(eris.ovov).conj().transpose(0,2,1,3)
    if cc.cc2:
        Woooo2 = eris.oooo.transpose(0,2,1,3)
        Woooo2 += np.einsum("lcki,jc->klij", eris_ovoo, t1)
        Woooo2 += np.einsum("kclj,ic->klij", eris_ovoo, t1)
        Woooo2 += np.einsum("kcld,ic,jd->klij", eris.ovov, t1, t1)
        t2new += np.einsum("klij,ka,lb->ijab", Woooo2, t1, t1)
        Wvvvv = np.einsum("kcbd,ka->abcd", eris_ovvv, -t1)
        Wvvvv = Wvvvv + Wvvvv.transpose(1,0,3,2)
        Wvvvv += eris.vvvv.transpose(0,2,1,3)
        t2new += np.einsum("abcd,ic,jd->ijab", Wvvvv, t1, t1)
        Lvv2 = fvv - np.einsum("kc,ka->ac", fov, t1)
        #Lvv2 -= np.diagflat(np.diag(fvv))
        tmp = np.einsum("ac,ijcb->ijab", Lvv2, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        Loo2 = foo + np.einsum("kc,ic->ki", fov, t1)
        #Loo2 -= np.diagflat(np.diag(foo))
        tmp = np.einsum("ki,kjab->ijab", Loo2, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
    else:
        Loo = imd.Loo(t1, t2, eris)
        Lvv = imd.Lvv(t1, t2, eris)

        Woooo = imd.cc_Woooo(t1, t2, eris)
        Wvoov = imd.cc_Wvoov(t1, t2, eris)
        Wvovo = imd.cc_Wvovo(t1, t2, eris)
        Wvvvv = imd.cc_Wvvvv(t1, t2, eris)

        tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
        t2new += np.einsum("klij,klab->ijab", Woooo, tau)
        t2new += np.einsum("abcd,ijcd->ijab", Wvvvv, tau)
        tmp = np.einsum("ac,ijcb->ijab", Lvv, t2)
        t2new += tmp + tmp.transpose(1,0,3,2)
        tmp = np.einsum("ki,kjab->ijab", Loo, t2)
        t2new -= tmp + tmp.transpose(1,0,3,2)
        tmp  = 2.*np.einsum("akic,kjcb->ijab", Wvoov, t2)
        tmp -=   np.einsum("akci,kjcb->ijab", Wvovo, t2)
        t2new += tmp + tmp.transpose(1,0,3,2)
        tmp = np.einsum("akic,kjbc->ijab", Wvoov, t2)
        t2new -= tmp + tmp.transpose(1,0,3,2)
        tmp = np.einsum("bkci,kjac->ijab", Wvovo, t2)
        t2new -= tmp + tmp.transpose(1,0,3,2)
    return t1new, t2new

class RCCSD(ccsd.CCSD):
    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False):
        return self.ccsd(t1, t2, eris, mbpt2)

    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        if mbpt2:
            raise NotImplementedError

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        return ccsd.CCSD.ccsd(self, t1, t2, eris)

    def ao2mo(self, mo_coeff=None):
        nmo = self.nmo
        nao = self.mo_coeff.shape[0]
        nmo_pair = nmo * (nmo+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
        mem_now = current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return _make_eris_incore(self, mo_coeff)
        elif getattr(self._scf, "with_df", None):
            raise NotImplementedError
        else:
            raise NotImplementedError

    update_amps = update_amps
    amplitude_equation = amplitude_equation

def _make_eris_incore(mycc, mo_coeff=None, ao2mofn=None):
    log = logger.new_logger(mycc)
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]

    if callable(ao2mofn):
        eri1 = ao2mofn(eris.mo_coeff).reshape([nmo]*4)
    else:
        eri1 = ao2mo.incore.full(mycc._scf._eri, eris.mo_coeff, compact=False)
    eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc]
    #eris.ooov = eri1[:nocc,:nocc,:nocc,nocc:]
    eris.ovoo = eri1[:nocc,nocc:,:nocc,:nocc]
    eris.ovov = eri1[:nocc,nocc:,:nocc,nocc:]
    eris.oovv = eri1[:nocc,:nocc,nocc:,nocc:]
    eris.ovvo = eri1[:nocc,nocc:,nocc:,:nocc]
    eris.ovvv = eri1[:nocc,nocc:,nocc:,nocc:]
    eris.vvvv = eri1[nocc:,nocc:,nocc:,nocc:]
    log.timer("CCSD integral transformation")
    del log
    return eris

class _ChemistsERIs(ccsd._ChemistsERIs):
    def get_ovvv(self, *slices):
        """To access a subblock of ovvv tensor"""
        if slices:
            return self.ovvv[slices]
        else:
            return self.ovvv


def restore_eri_s1(eri: Array, nao: int) -> Array:
    """Full ``(nao,)*4`` AO integral tensor from an ``s1``, ``s4`` or ``s8``
    integral array, as returned by :meth:`~pyscfad.gto.MoleLite.intor`.
    """
    nao_pair = nao * (nao+1) // 2
    if eri.size == nao**4:
        return eri.reshape([nao]*4)
    if eri.size == nao_pair * (nao_pair+1) // 2:
        eri = lib.unpack_tril(eri.ravel(), filltriu=lib.SYMMETRIC)
    if eri.size == nao_pair**2:
        return ao2mo.restore(1, eri.reshape(nao_pair, nao_pair), nao)
    raise NotImplementedError(
        f"eri of size {eri.size} is not an s1, s4 or s8 symmetrized "
        f"integral array of {nao} orbitals")

def _sort_mo(
    mo_coeff: Array,
    mo_occ: Array,
    mo_mask: Array,
) -> tuple[Array, Array]:
    """Reorder the MOs as ``[occupied, padding, virtual]``.

    The sort is stable, so each of the three groups keeps its energy order.
    """
    key = np.where(mo_occ > 0, 0, np.where(mo_mask, 2, 1))
    order = np.argsort(key)
    return mo_coeff[:,order], mo_mask[order]

class _ChemistsERIsLite:
    """MO integrals ``(pq|rs)`` of the lightweight CCSD path.

    Unlike :class:`_ChemistsERIs`, this is a plain container rather than a
    pytree: the solver closes over it instead of passing it across a
    :func:`jax.jit` boundary.

    Attributes:
        mo_mask: Mask flagging the real (non-padding) MOs, or ``None`` when
            the molecule is not padded.
    """
    def __init__(self):
        self.mo_coeff = None
        self.mo_mask = None
        self.nocc = None
        self.fock = None
        self.mo_energy = None
        self.e_hf = None

        self.oooo = None
        self.ovoo = None
        self.ovov = None
        self.oovv = None
        self.ovvo = None
        self.ovvv = None
        self.vvvv = None

    def get_ovvv(self, *slices):
        """To access a subblock of the ``ovvv`` tensor."""
        if slices:
            return self.ovvv[slices]
        return self.ovvv

def _make_eris_incore_lite(mycc, mo_coeff=None):
    """MO integrals of the lightweight path.

    The AO integrals are reused from ``mycc._scf._eri`` in whatever
    permutation symmetry the mean field stored them in
    (:meth:`~pyscfad.scf.hf_lite.SCFLite.get_jk` builds an ``s8`` array), so
    they -- and their derivatives -- are computed only once.
    """
    log = logger.new_logger(mycc)
    mf = mycc._scf
    mol = mycc.mol
    if mo_coeff is None:
        mo_coeff = mycc.mo_coeff

    eris = _ChemistsERIsLite()
    nocc = eris.nocc = mycc.nocc

    if getattr(mol, "ao_mask", None) is None:
        mo_mask = None
    else:
        # A padded molecule carries fake MOs, which SCFPad stores at the top of
        # the spectrum. A static nocc larger than the true occupation would
        # then pull real virtual orbitals into the occupied block; sorting the
        # fake MOs in between fills the extra occupied slots with orbitals
        # that enter no integral instead.
        mo_mask = mf.mo_mask(mf.mo_energy, mo_coeff)
        mo_coeff, mo_mask = _sort_mo(mo_coeff, mycc.mo_occ, mo_mask)
    eris.mo_coeff = mo_coeff
    eris.mo_mask = mo_mask

    # the Fock matrix and the HF energy are recomputed since the mean field
    # may not be fully converged
    dm = mf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
    h1e = mf.get_hcore(mol)
    vhf = mf.get_veff(mol, dm)
    fock_ao = mf.get_fock(h1e=h1e, vhf=vhf, dm=dm)
    eris.fock = mo_coeff.conj().T @ fock_ao @ mo_coeff
    eris.mo_energy = eris.fock.diagonal().real
    eris.e_hf = mf.energy_tot(dm, h1e, vhf)

    nao = mo_coeff.shape[0]
    eri_ao = mf._eri
    if eri_ao is None:
        eri_ao = mol.intor("int2e", aosym="s1")
    # TODO optimize ao2mo for higher permutation symmetries
    eri1 = ao2mo.incore.full(restore_eri_s1(eri_ao, nao), mo_coeff,
                             compact=False)
    eri_ao = None

    eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc]
    eris.ovoo = eri1[:nocc,nocc:,:nocc,:nocc]
    eris.ovov = eri1[:nocc,nocc:,:nocc,nocc:]
    eris.oovv = eri1[:nocc,:nocc,nocc:,nocc:]
    eris.ovvo = eri1[:nocc,nocc:,nocc:,:nocc]
    eris.ovvv = eri1[:nocc,nocc:,nocc:,nocc:]
    eris.vvvv = eri1[nocc:,nocc:,nocc:,nocc:]
    log.timer("CCSD integral transformation")
    del log
    return eris

def init_amps(mycc, eris) -> tuple[float, Array, Array]:
    """MP2 amplitudes, the initial guess of the CCSD iterations."""
    nocc = eris.nocc
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift
    eia, eijab = mo_energy_denominators(mo_e_o, mo_e_v, eris.mo_mask)

    t1 = eris.fock[:nocc,nocc:] / eia
    eris_ovov = eris.ovov
    t2 = eris_ovov.transpose(0,2,1,3).conj() / eijab
    emp2  = 2 * np.einsum("ijab,iajb", t2, eris_ovov)
    emp2 -=     np.einsum("jiab,iajb", t2, eris_ovov)
    return emp2.real, t1, t2

def _amp_norm(t1: Array, t2: Array) -> Array:
    return np.sqrt(np.vdot(t1, t1).real + np.vdot(t2, t2).real)

def _solve_amps(
    mycc,
    t1: Array,
    t2: Array,
    eris: Any,
    conv_tol: float,
    conv_tol_normt: float,
) -> tuple[tuple[Array, Array], Array]:
    """Iterate the amplitude equations to convergence.

    Notes:
        The iteration runs in a :func:`jax.lax.while_loop`, so the amplitude
        mixer is carried as part of the loop state.
    """
    log = logger.new_logger(mycc)

    def cond_fun(value):
        cycle, de, normt = value[:3]
        return (cycle < mycc.max_cycle) & ((abs(de) > conv_tol) | (normt > conv_tol_normt))

    def body_fun(value):
        cycle, _, _, t1, t2, e_corr, diis = value
        t1new, t2new = mycc.update_amps(t1, t2, eris)
        normt = _amp_norm(t1new-t1, t2new-t2)
        alpha = mycc.iterative_damping
        if alpha < 1. and alpha > 0.:
            t1new = (1.-alpha) * t1 + alpha * t1new
            t2new = (1.-alpha) * t2 + alpha * t2new
        (t1, t2), diis = mycc.run_diis((t1new, t2new), (t1, t2), diis)
        e_last, e_corr = e_corr, mycc.energy(t1, t2, eris)
        log.info("cycle = %d  E_corr(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g",
                 cycle+1, e_corr, e_corr-e_last, normt)
        return cycle+1, e_corr-e_last, normt, t1, t2, e_corr, diis

    diis = mycc.init_diis((t1, t2))
    e_corr = mycc.energy(t1, t2, eris)
    log.info("Init E_corr(CCSD) = %.15g", e_corr)
    big = np.asarray(1e3, dtype=np.floatx)
    init_val = (0, big, big, t1, t2, e_corr, diis)
    _, de, normt, t1, t2, _, _ = while_loop(cond_fun, body_fun, init_val)
    conv = (abs(de) <= conv_tol) & (normt <= conv_tol_normt)
    del log
    return (t1, t2), conv

def _solve_amps_implicit(
    mycc,
    t1: Array,
    t2: Array,
    eris: Any,
    conv_tol: float,
    conv_tol_normt: float,
) -> tuple[Array, Array, Array]:
    """Converged amplitudes, differentiated through the amplitude equations.

    The forward solve is handed to :func:`jax.lax.custom_root`, which
    recovers the derivative from the fixed-point condition
    ``update_amps(t1, t2) - (t1, t2) == 0`` instead of unrolling the
    iterations.
    """
    def oracle(fn, amps):
        del fn
        amps, conv = _solve_amps(mycc, amps[0], amps[1], eris,
                                 conv_tol, conv_tol_normt)
        # the auxiliary output is given a symbolic zero tangent, which JAX
        # only builds for a floating-point aval
        return amps, np.asarray(conv, dtype=np.floatx)

    def root_fn(amps):
        t1new, t2new = mycc.update_amps(amps[0], amps[1], eris)
        return t1new - amps[0], t2new - amps[1]

    # FIXME restore to use jax gmres once issue
    # (https://github.com/jax-ml/jax/issues/33872) is fixed
    solver = partial(gmres_const_atol,
                     tol=mycc._conv_tol_implicit_diff,
                     atol=mycc._conv_tol_implicit_diff,
                     maxiter=mycc._max_cycle_implicit_diff,
                     solve_method="batched",
                     restart=mycc._restart_implicit_diff)
    def tangent_solve(g, amps_bar):
        return solver(g, amps_bar)[0]

    (t1, t2), conv = custom_root(root_fn, (t1, t2), oracle, tangent_solve,
                                 has_aux=True)
    return t1, t2, conv.astype(bool)

def kernel(mycc, eris=None, t1=None, t2=None):
    log = logger.new_logger(mycc)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if t1 is None or t2 is None:
        emp2, t1_guess, t2_guess = mycc.init_amps(eris)
        mycc.emp2 = emp2
        log.info("Init t2, MP2 energy = %.15g  E_corr(MP2) %.15g",
                 eris.e_hf + emp2, emp2)
        if t1 is None:
            t1 = t1_guess
        if t2 is None:
            t2 = t2_guess

    t1, t2, conv = _solve_amps_implicit(mycc, t1, t2, eris,
                                        mycc.conv_tol, mycc.conv_tol_normt)
    # the auxiliary output of custom_root carries no tangent, so the
    # correlation energy is recomputed from the (differentiable) amplitudes
    e_corr = mycc.energy(t1, t2, eris)
    log.timer("CCSD")
    del log
    return conv, e_corr, t1, t2

class RCCSDLite:
    """Restricted CCSD for the lightweight molecule and mean-field objects.

    Parameters:
        mf: Converged :class:`~pyscfad.scf.hf_lite.SCFLite` (or
            :class:`~pyscfad.ml.scf.SCFPad`) mean field.
        mo_coeff: MO coefficients. Defaults to ``mf.mo_coeff``.
        mo_occ: MO occupations. Defaults to ``mf.mo_occ``.
        nocc: Number of doubly occupied orbitals. Static; inferred from the
            molecular electron count when not given, which requires that
            count to be concrete. For a padded molecule it may exceed the
            true occupation -- as a batch of padded molecules requires --
            and the extra occupied slots are then filled with fake orbitals.

    Attributes:
        diis: Amplitude mixer, ``'diis'`` for Pulay's DIIS (the default),
            ``'anderson'`` for Anderson mixing, or ``None`` for the bare
            iterations.
        conv_tol_implicit_diff: Convergence threshold of the GMRES solve of
            the implicit derivative.
        max_cycle_implicit_diff: Maximum number of GMRES iterations.
        restart_implicit_diff: Krylov subspace size of the GMRES solve.
    """
    max_cycle: int = 50
    conv_tol: float = 1e-7
    conv_tol_normt: float = 1e-5
    iterative_damping: float = 1.
    level_shift: float = 0.
    cc2: bool = False

    diis: str | None = "diis"
    diis_space: int = 6
    diis_damp: float = 0.
    diis_start_cycle: int = 0

    _conv_tol_implicit_diff: float = 1e-6
    _max_cycle_implicit_diff: int = 40
    _restart_implicit_diff: int = 20

    def __init__(
        self,
        mf: SCFLite,
        mo_coeff: Array | None = None,
        mo_occ: Array | None = None,
        nocc: int | None = None,
        **kwargs,
    ):
        self._scf = mf
        self.mol = mf.mol
        self.verbose = mf.verbose
        self.stdout = getattr(mf, "stdout", sys.stdout)
        self.mo_coeff = mf.mo_coeff if mo_coeff is None else mo_coeff
        self.mo_occ = mf.mo_occ if mo_occ is None else mo_occ
        self._nocc = nocc
        if self.mo_coeff is None or self.mo_occ is None:
            raise ValueError("The mean-field object has no MOs. Run its "
                             "kernel first, or pass mo_coeff and mo_occ.")

        self.converged = False
        self.e_hf = None
        self.e_corr = None
        self.emp2 = None
        self.t1 = None
        self.t2 = None

        for key, val in kwargs.items():
            setattr(self, key, val)

    @property
    def nocc(self) -> int:
        """Number of doubly occupied orbitals. A static quantity."""
        if self._nocc is not None:
            return self._nocc
        try:
            return int(self.mol.tot_electrons()) // 2
        except TypeError as err:
            raise TypeError(
                "'nocc' cannot be inferred from a traced electron count. "
                "Pass it explicitly, e.g. RCCSDLite(mf, nocc=nocc). A padded "
                "molecule needs it whenever the atomic numbers are traced; "
                "use a value no smaller than the largest occupation of the batch."
            ) from err

    @nocc.setter
    def nocc(self, n: int | None):
        self._nocc = n

    @property
    def nmo(self) -> int:
        return self.mo_coeff.shape[-1]

    @property
    def nvir(self) -> int:
        return self.nmo - self.nocc

    @property
    def e_tot(self) -> float:
        return self.e_hf + self.e_corr

    def dump_flags(self, verbose: int | None = None):
        pass

    def ao2mo(self, mo_coeff: Array | None = None) -> _ChemistsERIsLite:
        return _make_eris_incore_lite(self, mo_coeff)

    def init_diis(self, amps: tuple[Array, Array]) -> Any:
        """Amplitude mixer.

        Notes:
            Both mixers extrapolate an arbitrary pytree, so the amplitudes
            are mixed as the ``(t1, t2)`` tuple rather than packed into a
            vector. Unlike :class:`pyscfad.lib.diis.DIIS`, they are pytrees
            themselves and can therefore be carried through
            :func:`jax.lax.while_loop`.
        """
        if self.diis is None or self.diis is False:
            return None
        if isinstance(self.diis, str):
            key = self.diis.lower()
            if key == "diis":
                return DIISLite(amps, space=self.diis_space)
            if key == "anderson":
                return Anderson(
                    amps,
                    space=self.diis_space,
                    damp=self.diis_damp,
                    start_cycle=self.diis_start_cycle,
                )
        raise NotImplementedError(f"Unsupported diis {self.diis!r}")

    def run_diis(
        self,
        amps: tuple[Array, Array],
        amps_last: tuple[Array, Array],
        diis: Any,
    ) -> tuple[tuple[Array, Array], Any]:
        if diis is None:
            return amps, diis
        if isinstance(diis, Anderson):
            # Anderson mixing takes the previous vector, whereas DIIS forms
            # the error vector against it itself (they are the same vector:
            # amps_last is the previous extrapolation)
            return diis.update(amps, amps_last), diis
        return diis.update(amps), diis

    def ccsd(self, t1=None, t2=None, eris=None):
        self.dump_flags()
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        self.e_hf = eris.e_hf

        self.converged, self.e_corr, self.t1, self.t2 = kernel(self, eris, t1, t2)
        self._finalize()
        return self.e_corr, self.t1, self.t2

    def kernel(self, t1=None, t2=None, eris=None):
        return self.ccsd(t1, t2, eris)

    def _finalize(self):
        logger.note(self, "E(%s) = %.16g  E_corr = %.16g  converged = %s",
                    self.__class__.__name__, self.e_tot, self.e_corr,
                    self.converged)
        return self

    energy = energy
    init_amps = init_amps
    update_amps = update_amps
    amplitude_equation = amplitude_equation
