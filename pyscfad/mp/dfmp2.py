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

from functools import partial
import ctypes

import numpy
import jax
from jax import custom_vjp
from jax.custom_derivatives import SymbolicZero

from pyscf import __config__
from pyscf import lib as pyscf_lib
from pyscf.lib import direct_sum, current_memory
from pyscf.ao2mo import _ao2mo as pyscf_ao2mo

from pyscfad import config
from pyscfad import numpy as np
from pyscfad.ops import vmap
from pyscfad.lib import logger
from pyscfad.lib import _cotangent_buffer
from pyscfad.ao2mo import _ao2mo, _ao2mo_opt
from pyscfad.mp import mp2

try:
    from pyscfadlib import libmp_vjp
except ImportError:
    # NOTE MP2_contract_d_vjp needs pyscfadlib>0.3.4
    libmp_vjp = None

libmp = pyscf_lib.load_library('libmp')

WITH_T2 = getattr(__config__, 'mp_dfmp2_with_t2', True)

def _contract(Lov, mo_energy, nocc, nvir, with_t2=True):
    def body(Lv, Lov, ea, eia):
        gi = np.einsum('la,ljb->jab', Lv, Lov)
        t2i = gi / (eia[:,:,None] + ea[None,None,:])
        ei = np.einsum('jab,jab', t2i, gi) * 2 - np.einsum('jab,jba', t2i, gi)
        return ei, t2i

    Lov = Lov.reshape((-1, nocc, nvir))
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    e, t2 = vmap(body, in_axes=(1,None,0,None),
                 signature='(l,a),(b)->(),(j,a,b)')(Lov, Lov, eia, eia)
    if not with_t2:
        t2 = None
    emp2 = e.sum().real
    return emp2, t2

@partial(custom_vjp, nondiff_argnums=(2,3,4))
def _contract_opt(Lov, mo_energy, nocc, nvir, with_t2=True):
    if with_t2:
        t2 = numpy.empty((nocc,nocc,nvir,nvir))
    else:
        t2 = None
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    emp2 = 0
    for i in range(nocc):
        buf = numpy.dot(Lov[:,i*nvir:(i+1)*nvir].T,
                        Lov).reshape((nvir,nocc,nvir))
        gi = buf.transpose(1,0,2)
        t2i = gi / direct_sum('jb+a->jba', eia, eia[i])
        emp2 += numpy.einsum('jab,jab', t2i, gi) * 2
        emp2 -= numpy.einsum('jab,jba', t2i, gi)
        if with_t2:
            t2[i] = t2i
    return emp2, t2

def _contract_opt_fwd(Lov, mo_energy, nocc, nvir, with_t2):
    emp2, t2 = _contract_opt(Lov, mo_energy, nocc, nvir, with_t2)
    return (emp2, t2), (Lov, mo_energy, t2)

def _contract_opt_bwd(nocc, nvir, with_t2,
                      res, ybar):
    Lov, mo_energy, t2 = res
    emp2_bar = ybar[0]
    if with_t2:
        t2 = numpy.asarray(t2)
        t2_bar = numpy.asarray(ybar[1])

    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    Lov_bar = numpy.zeros_like(Lov)
    mo_energy_bar = numpy.zeros_like(mo_energy)
    for i in range(nocc):
        ejab = direct_sum('jb+a->jba', eia, eia[i])
        if with_t2:
            t2i = t2[i]
            gi = t2i * ejab
        else:
            gi = numpy.dot(Lov[:,i*nvir:(i+1)*nvir].T,
                           Lov).reshape((nvir,nocc,nvir))
            gi = gi.transpose(1,0,2)
            t2i = gi / ejab

        buf = emp2_bar * t2i
        gi_bar = 2 * buf - buf.transpose(0,2,1)
        buf = emp2_bar * gi
        t2i_bar = 2 * buf - buf.transpose(0,2,1)
        if with_t2:
            t2i_bar += t2_bar[i]
        gi_bar += t2i_bar / ejab

        buf = gi_bar.transpose(1,0,2).reshape(nvir,-1)
        Lov_bar += numpy.dot(Lov[:,i*nvir:(i+1)*nvir], buf)
        Lov_bar[:,i*nvir:(i+1)*nvir] += numpy.dot(Lov, buf.T)
        gi_bar = None

        buf = -gi * t2i_bar / (ejab * ejab)
        ejab = gi = t2i_bar = None
        mo_energy_bar[i] += numpy.sum(buf)
        mo_energy_bar[:nocc] += numpy.sum(buf.reshape(nocc,-1), axis=-1)
        mo_energy_bar[nocc:] -= numpy.sum(buf.reshape(-1,nvir), axis=0)
        mo_energy_bar[nocc:] -= numpy.sum(buf.transpose(1,0,2).reshape(nvir,-1), axis=-1)
        buf = None
    return (Lov_bar, mo_energy_bar)

_contract_opt.defvjp(_contract_opt_fwd, _contract_opt_bwd)

def _contract_scan(Lov, mo_energy, nocc, nvir, with_t2=True):
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    emp2 = mp2.E_CORR_MP2()

    @jax.checkpoint
    def _fn(emp2, x):
        La, ea = x
        gi = np.dot(La.T, Lov).reshape((nvir,nocc,nvir))
        gi = gi.transpose(1,0,2)
        t2i = gi / (eia[:,:,None] + ea[None,None,:])
        edi = np.einsum('jab,jab', t2i, gi) * 2
        exi = -np.einsum('jab,jba', t2i, gi)
        emp2.e_corr_ss += edi * 0.5 + exi
        emp2.e_corr_os += edi * 0.5
        if not with_t2:
            t2i = None
        return emp2, t2i

    emp2, t2 = jax.lax.scan(_fn, emp2, (Lov.reshape((-1,nocc,nvir)).transpose(1,0,2), eia))
    emp2.e_corr_ss = emp2.e_corr_ss.real
    emp2.e_corr_os = emp2.e_corr_os.real
    emp2.e_corr = emp2.e_corr_ss + emp2.e_corr_os
    if not with_t2:
        t2 = None
    return emp2, t2

def _to_iaL(Lov, nocc, nvir):
    # (nocc, nvir, naux), the layout of MP2_contract_d
    return pyscf_lib.transpose(numpy.asarray(Lov)).reshape(nocc, nvir, -1)

def _ed_ex(iaL, mo_energy, nocc, nvir, t2=None):
    '''(ed, ex) of MP2_contract_d, and the amplitudes written to t2,
    (nocc, nocc, nvir, nvir) in C order, if given.'''
    naux = iaL.shape[-1]
    e_occ = mo_energy[:nocc]
    e_vir = mo_energy[nocc:]
    moeoo = numpy.asarray(e_occ[:,None] + e_occ, order='C')
    moevv = numpy.asarray(e_vir[:,None] + e_vir, order='C')
    ed = numpy.zeros(1)
    ex = numpy.zeros(1)
    libmp.MP2_contract_d(
        ed.ctypes.data_as(ctypes.c_void_p),
        ex.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(1),
        iaL.ctypes.data_as(ctypes.c_void_p),
        iaL.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(0), ctypes.c_int(0),
        ctypes.c_int(nocc), ctypes.c_int(nocc),
        ctypes.c_int(nocc), ctypes.c_int(nvir), ctypes.c_int(naux),
        moeoo.ctypes.data_as(ctypes.c_void_p),
        moevv.ctypes.data_as(ctypes.c_void_p),
        _c_ptr(t2), ctypes.c_int(0),
    )
    return ed[0], ex[0]

def _c_ptr(a):
    if a is None:
        return pyscf_lib.c_null_ptr()
    return a.ctypes.data_as(ctypes.c_void_p)

def _contract_d_vjp(iaL, mo_energy, nocc, nvir, ed_bar, ex_bar,
                    iaL_bar=None, e_bar=None, with_energy=False,
                    t2=None, t2_bar=None):
    '''VJP of (ed, ex) of MP2_contract_d for the cotangents (ed_bar, ex_bar),
    and of the amplitudes for t2_bar if given, added to iaL_bar and e_bar if
    given. If with_energy, also returns ed and ex, and writes the amplitudes
    to t2 if given, computed in the same pass (MP2_contract_d_energy_vjp).
    '''
    naux = iaL.shape[-1]
    if iaL_bar is None:
        iaL_bar = numpy.zeros_like(iaL)
    if e_bar is None:
        e_bar = numpy.zeros_like(mo_energy)
    e_vir_bar = e_bar[nocc:]
    args = (
        iaL_bar.ctypes.data_as(ctypes.c_void_p),
        e_bar.ctypes.data_as(ctypes.c_void_p),
        e_vir_bar.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(ed_bar),
        ctypes.c_double(ex_bar),
        iaL.ctypes.data_as(ctypes.c_void_p),
        mo_energy.ctypes.data_as(ctypes.c_void_p),
        mo_energy[nocc:].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nocc), ctypes.c_int(nvir), ctypes.c_int(naux),
    )
    if with_energy or t2 is not None or t2_bar is not None:
        e = numpy.zeros(2)
        if t2_bar is not None:
            t2_bar = numpy.asarray(t2_bar, dtype=numpy.float64, order='C')
            assert t2_bar.shape == (nocc, nocc, nvir, nvir)
        libmp_vjp.MP2_contract_d_energy_vjp(_c_ptr(e), _c_ptr(t2), _c_ptr(t2_bar),
                                            *args)
        if with_energy:
            return e[0], e[1], iaL_bar, e_bar
    else:
        libmp_vjp.MP2_contract_d_vjp(*args)
    return iaL_bar, e_bar

def _iaL_from_cderi(eri, mo_coeff, nocc, nvir):
    Lov = pyscf_ao2mo.nr_e2(numpy.asarray(eri, order='C'),
                            numpy.asarray(mo_coeff, order='F'),
                            (0, nocc, nocc, nocc+nvir), aosym='s2')
    return _to_iaL(Lov, nocc, nvir)

@partial(custom_vjp, nondiff_argnums=(3,4,5))
def _contract_d_cderi(eri, mo_coeff, mo_energy, nocc, nvir, with_t2=False):
    '''(ed, ex) of MP2_contract_d from the DF tensor eri, and the amplitudes
    t2, (nocc, nocc, nvir, nvir), if with_t2.

    In reverse mode, the forward rule computes in the same pass the VJP for
    the cotangent (2, 1) of e_corr = 2*ed + ex, the one of e_tot, with three
    dgemm per pair instead of one for the energy and three for the VJP.
    For other cotangents, and that of t2, the backward rule adds the rest.
    Only this VJP, the size of Lov, is kept between the passes, and the
    cotangent of eri goes to the buffer of the DF tensor
    (see pyscfad.lib._cotangent_buffer).
    '''
    iaL = _iaL_from_cderi(eri, mo_coeff, nocc, nvir)
    mo_energy = numpy.asarray(mo_energy, dtype=numpy.float64, order='C')
    if with_t2:
        t2 = numpy.empty((nocc,nocc,nvir,nvir))
        return _ed_ex(iaL, mo_energy, nocc, nvir, t2) + (t2,)
    return _ed_ex(iaL, mo_energy, nocc, nvir)

def _contract_d_cderi_fwd(eri, mo_coeff, mo_energy, nocc, nvir, with_t2):
    perturbed = (eri.perturbed, mo_coeff.perturbed, mo_energy.perturbed)
    eri = eri.value
    mo_coeff = mo_coeff.value
    mo_energy = numpy.asarray(mo_energy.value, dtype=numpy.float64, order='C')
    iaL = _iaL_from_cderi(eri, mo_coeff, nocc, nvir)
    t2 = numpy.empty((nocc,nocc,nvir,nvir)) if with_t2 else None
    if hasattr(libmp_vjp, 'MP2_contract_d_energy_vjp'):
        ed, ex, iaL_bar, e_bar = _contract_d_vjp(iaL, mo_energy, nocc, nvir,
                                                 2., 1., with_energy=True, t2=t2)
        e_corr_vjp = (iaL_bar, e_bar)
    else:
        ed, ex = _ed_ex(iaL, mo_energy, nocc, nvir, t2)
        e_corr_vjp = None
    out = (ed, ex, t2) if with_t2 else (ed, ex)
    return out, (eri, mo_coeff, mo_energy, perturbed, e_corr_vjp)

def _contract_d_cderi_bwd(nocc, nvir, with_t2, res, ybar):
    log = logger.new_logger(verbose=4)
    eri, mo_coeff, mo_energy, perturbed, e_corr_vjp = res
    ed_bar, ex_bar = [0. if isinstance(y, SymbolicZero) else float(y)
                      for y in ybar[:2]]
    t2_bar = None
    if with_t2 and not isinstance(ybar[2], SymbolicZero):
        t2_bar = ybar[2]
    if ed_bar == 0 and ex_bar == 0 and t2_bar is None:
        return (None, None, None)

    iaL_bar = e_bar = None
    scale = 1.
    if e_corr_vjp is not None:
        # (ed_bar, ex_bar) = ex_bar * (2, 1) + (ed_bar - 2*ex_bar, 0)
        iaL_bar, e_bar = e_corr_vjp
        scale = ex_bar
        ed_bar, ex_bar = ed_bar - 2 * ex_bar, 0.
        if ed_bar != 0 or t2_bar is not None:
            iaL_bar = iaL_bar * scale
            e_bar = e_bar * scale
            scale = 1.
    if ed_bar != 0 or ex_bar != 0 or t2_bar is not None:
        iaL = _iaL_from_cderi(eri, mo_coeff, nocc, nvir)
        iaL_bar, e_bar = _contract_d_vjp(iaL, mo_energy, nocc, nvir,
                                         ed_bar, ex_bar, iaL_bar, e_bar,
                                         t2_bar=t2_bar)
        iaL = None

    naux = iaL_bar.shape[-1]
    Lov_bar = pyscf_lib.transpose(iaL_bar.reshape(nocc*nvir, naux))
    iaL_bar = None
    if scale != 1:
        Lov_bar *= scale
        e_bar = e_bar * scale

    eri_bar_buf = _cotangent_buffer.get(eri) if perturbed[0] else None
    eri_bar, mo_coeff_bar = _ao2mo_opt.nr_e2_vjp(
        eri, mo_coeff, (0, nocc, nocc, nocc+nvir), Lov_bar,
        aosym='s2', eri_bar=eri_bar_buf)
    if eri_bar_buf is not None or not perturbed[0]:
        eri_bar = None
    log.timer('_contract_d_cderi_bwd')
    del log
    return (eri_bar,
            mo_coeff_bar if perturbed[1] else None,
            e_bar if perturbed[2] else None)

_contract_d_cderi.defvjp(_contract_d_cderi_fwd, _contract_d_cderi_bwd,
                         symbolic_zeros=True)

def _check_incore_memory(mp, nocc, nvir, with_t2=WITH_T2):
    naux = mp.with_df.get_naoaux()
    mem_incore = (naux*nocc*nvir + nocc*nvir*nvir*2) * 8 / 1e6
    if with_t2:
        mem_incore += (nocc*nvir)**2 * 8 / 1e6
    mem_now = current_memory()[0]
    if mem_incore + mem_now >= mp.max_memory and not mp.mol.incore_anyway:
        raise RuntimeError(f'{mem_incore+mem_now} MB of memory is needed.')

def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=None):
    if mo_energy is not None or mo_coeff is not None:
        assert (mp.frozen == 0 or mp.frozen is None)

    if eris is None:
        eris = mp.ao2mo(mo_coeff)
    if mo_energy is None:
        mo_energy = eris.mo_energy
    if mo_coeff is None:
        mo_coeff = eris.mo_coeff

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    #naux = mp.with_df.get_naoaux()

    # loop_ao2mo fused, see _contract_d_cderi; the amplitudes and their
    # cotangent need MP2_contract_d_energy_vjp
    fused = (config.moleintor_opt and libmp_vjp is not None
             and (not with_t2 or hasattr(libmp_vjp, 'MP2_contract_d_energy_vjp')))

    log = logger.new_logger(mp)
    if fused:
        _check_incore_memory(mp, nocc, nvir, with_t2)
        out = _contract_d_cderi(mp.with_df._cderi, mo_coeff, mo_energy,
                                nocc, nvir, with_t2)
        ed, ex = out[:2]
        t2 = out[2] if with_t2 else None
        emp2 = mp2.E_CORR_MP2(e_corr=ed*2+ex, e_corr_ss=ed+ex, e_corr_os=ed)
    else:
        Lov = mp.loop_ao2mo(mo_coeff, nocc, with_t2)
        emp2, t2 = _contract_scan(Lov, mo_energy, nocc, nvir, with_t2)

    log.timer(f'{mp.__class__.__name__} kernel')
    del log
    return emp2, t2

class DFRMP2(mp2.RMP2):
    _dynamic_attr = _keys = {'with_df'}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        super().__init__(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            raise KeyError('The mean-field object has no density fitting.')

    def ao2mo(self, mo_coeff=None):
        eris = mp2._ChemistsERIs()
        eris._common_init_(self, mo_coeff)
        return eris

    def loop_ao2mo(self, mo_coeff, nocc, with_t2=WITH_T2):
        # NOTE return the whole 3c integral for now
        nao, nmo = mo_coeff.shape
        nvir = nmo - nocc
        ijslice = (0, nocc, nocc, nmo)

        _check_incore_memory(self, nocc, nvir, with_t2)
        eri1 = self.with_df._cderi
        Lov = _ao2mo.nr_e2(eri1, mo_coeff, ijslice, aosym='s2')
        return Lov

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        if self.verbose >= logger.WARN:
            self.check_sanity()

        self.dump_flags()

        self.e_hf = self.get_e_hf(mo_coeff=mo_coeff)

        if eris is None:
            eris = self.ao2mo(mo_coeff)

        if self._scf.converged:
            e_corr, self.t2 = self.init_amps(mo_energy, mo_coeff, eris, with_t2)
        else:
            raise NotImplementedError

        self.e_corr_ss = e_corr.e_corr_ss
        self.e_corr_os = e_corr.e_corr_os
        self.e_corr = e_corr.e_corr
        self._finalize()
        return self.e_corr, self.t2

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

MP2 = DFRMP2

del WITH_T2
