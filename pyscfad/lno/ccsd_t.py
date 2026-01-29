# Copyright 2023-2026 The PySCFAD Authors
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

'''Impurity (T) correction.
'''

from functools import partial
import ctypes
import numpy
from jax import custom_vjp

from pyscf.lib import (
    logger,
    prange,
    unpack_tril,
    num_threads,
    current_memory,
)

from pyscfad import numpy as np
from pyscfad.tools import timer
from pyscfadlib import libcc_vjp as libcc

def kernel(mycc, eris, ulo, t1=None, t2=None, verbose=logger.NOTE):
    log = logger.new_logger(mycc, verbose)
    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2

    mat = np.dot(ulo.T, ulo)
    assert mat.shape[0] == t1.shape[0]

    nocc, nvir = t1.shape
    t1T = t1.T
    t2T = t2.transpose(2,3,1,0)
    mo_energy = eris.mo_energy
    fvo = eris.fock[nocc:,:nocc]
    ovoo = eris.ovoo
    ovov = eris.ovov
    ovvv = eris.ovvv

    et = _ccsd_t_energy(mat, t1T, t2T, mo_energy, fvo,
                        ovoo, ovov, ovvv, mycc.max_memory)

    log.timer('CCSD(T)')
    log.note('CCSD(T) correction = %.15g', et)
    del log
    return et

def get_ovvv(ovvv, *slices):
    ovw = numpy.asarray(ovvv[slices])
    nocc, nvir, nvir_pair = ovw.shape
    ovvv = unpack_tril(ovw.reshape(nocc*nvir,nvir_pair))
    nvir1 = ovvv.shape[2]
    # pylint: disable=too-many-function-args
    return ovvv.reshape(nocc,nvir,nvir1,nvir1)

@partial(custom_vjp, nondiff_argnums=(8,))
def _ccsd_t_energy(mat, t1T, t2T, mo_energy, fvo,
                   ovoo, ovov, ovvv, max_memory):
    nvir, nocc = t1T.shape
    nmo = nocc + nvir

    mat = numpy.asarray(mat, order='C')
    t1T = numpy.asarray(t1T, order='C')
    t2T = numpy.asarray(t2T, order='C')
    mo_energy = numpy.asarray(mo_energy, order='C')
    fvo = numpy.asarray(fvo, order='C')

    vooo = numpy.asarray(ovoo).conj().transpose(1,0,3,2)
    vooo = numpy.asarray(vooo, order='C')

    vvop = numpy.empty((nvir,nvir,nocc,nmo))
    vvop[:,:,:,:nocc] = numpy.asarray(ovov).conj().transpose(1,3,0,2)
    vvop[:,:,:,nocc:] = get_ovvv(ovvv).conj().transpose(1,3,0,2)
    vvop = numpy.asarray(vvop, order='C')

    drv = libcc.lnoccsdt_contract
    et_sum = numpy.zeros(1, dtype=float)
    def contract(a0, a1, cache):
        drv(et_sum.ctypes.data_as(ctypes.c_void_p),
            mat.ctypes.data_as(ctypes.c_void_p),
            mo_energy.ctypes.data_as(ctypes.c_void_p),
            t1T.ctypes.data_as(ctypes.c_void_p),
            t2T.ctypes.data_as(ctypes.c_void_p),
            vooo.ctypes.data_as(ctypes.c_void_p),
            fvo.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nocc), ctypes.c_int(nvir),
            ctypes.c_int(a0), ctypes.c_int(a1),
            cache.ctypes.data_as(ctypes.c_void_p))

    mem_now = current_memory()[0]
    max_memory = max(0, max_memory - mem_now)
    cache_size = (nocc**3*3+nocc*nvir*2+2)*num_threads()+nvir**2*(nvir+1)/2*7+nocc**3*3
    if max_memory*1e6/8 < cache_size:
        raise RuntimeError(f'{cache_size*8/1e6} MB more memory is needed.')
    contract(0, nvir, vvop)

    et_sum *= 2. / 3.
    et = et_sum[0].real
    return et

def _ccsd_t_energy_fwd(mat, t1T, t2T, mo_energy, fvo,
                       ovoo, ovov, ovvv, max_memory):
    et = _ccsd_t_energy(mat, t1T, t2T, mo_energy, fvo,
                        ovoo, ovov, ovvv, max_memory)
    return et, (mat, t1T, t2T, mo_energy, fvo, ovoo, ovov, ovvv)

def _ccsd_t_energy_bwd(max_memory, res, et_bar):
    mytimer = timer.Timer()

    mat, t1T, t2T, mo_energy, fvo, ovoo, ovov, ovvv = res

    nvir, nocc = t1T.shape
    nmo = nocc + nvir

    et_bar *= 2. / 3.

    mat = numpy.asarray(mat, order='C')
    mat_bar = numpy.zeros_like(mat)

    t1T = numpy.asarray(t1T, order='C')
    t1T_bar = numpy.zeros_like(t1T)
    t2T = numpy.asarray(t2T, order='C')
    t2T_bar = numpy.zeros_like(t2T)

    mo_energy = numpy.asarray(mo_energy, order='C')
    mo_energy_bar = numpy.zeros_like(mo_energy)
    fvo = numpy.asarray(fvo, order='C')
    fvo_bar = numpy.zeros_like(fvo)

    vooo = numpy.asarray(ovoo).conj().transpose(1,0,3,2)
    vooo = numpy.asarray(vooo, order='C')
    vooo_bar = numpy.zeros_like(vooo)

    vvop = numpy.empty((nvir,nvir,nocc,nmo))
    vvop[:,:,:,:nocc] = numpy.asarray(ovov).conj().transpose(1,3,0,2)
    vvop[:,:,:,nocc:] = get_ovvv(ovvv).conj().transpose(1,3,0,2)
    vvop = numpy.asarray(vvop, order='C')
    vvop_bar = numpy.zeros_like(vvop)

    drv = libcc.lnoccsdt_energy_vjp
    def contract(a0, a1, b0, b1, cache, cache_bar):
        cache_row_a, cache_col_a, cache_row_b, cache_col_b = cache
        cache_row_a_bar, cache_col_a_bar, cache_row_b_bar, cache_col_b_bar = cache_bar
        drv(mat.ctypes.data_as(ctypes.c_void_p),
            mo_energy.ctypes.data_as(ctypes.c_void_p),
            t1T.ctypes.data_as(ctypes.c_void_p),
            t2T.ctypes.data_as(ctypes.c_void_p),
            vooo.ctypes.data_as(ctypes.c_void_p),
            fvo.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(et_bar),
            ctypes.c_int(nocc), ctypes.c_int(nvir),
            ctypes.c_int(a0), ctypes.c_int(a1),
            ctypes.c_int(b0), ctypes.c_int(b1),
            cache_row_a.ctypes.data_as(ctypes.c_void_p),
            cache_col_a.ctypes.data_as(ctypes.c_void_p),
            cache_row_b.ctypes.data_as(ctypes.c_void_p),
            cache_col_b.ctypes.data_as(ctypes.c_void_p),
            mat_bar.ctypes.data_as(ctypes.c_void_p),
            mo_energy_bar.ctypes.data_as(ctypes.c_void_p),
            t1T_bar.ctypes.data_as(ctypes.c_void_p),
            t2T_bar.ctypes.data_as(ctypes.c_void_p),
            vooo_bar.ctypes.data_as(ctypes.c_void_p),
            fvo_bar.ctypes.data_as(ctypes.c_void_p),
            cache_row_a_bar.ctypes.data_as(ctypes.c_void_p),
            cache_col_a_bar.ctypes.data_as(ctypes.c_void_p),
            cache_row_b_bar.ctypes.data_as(ctypes.c_void_p),
            cache_col_b_bar.ctypes.data_as(ctypes.c_void_p))

    mem_now = current_memory()[0]
    max_memory = max(0, max_memory - mem_now)

    min_memory = nocc**3*3+nvir*nocc*2
    min_memory+= (nmo*(nocc+1) + nvir*nocc*(nvir*nocc+1) + (nvir+6)*nocc**3 + 2) * (num_threads()-1)
    min_memory*= 8./1e6
    bufsize = (max_memory - min_memory)*1e6/8/num_threads()/(nocc*nmo*nvir+nvir)/2
    bufsize *= .8
    if bufsize < 8:
        mem_need = min_memory + 160.*(nocc*nmo*nvir+nvir)*num_threads()/1e6
        raise RuntimeError(f'_ccsd_t_energy_vjp: at least {mem_need} MB of more memory needed.')
    bufsize = int(max(8, bufsize))

    for a0, a1 in reversed(list(prange(0, nvir, bufsize))):
        cache_row_a = numpy.asarray(vvop[a0:a1,:], order='C')
        cache_row_a_bar = numpy.zeros_like(cache_row_a)
        if (a0, a1) == (0, nvir):
            cache_col_a = cache_row_a
            cache_col_a_bar = cache_row_a_bar
        else:
            cache_col_a = numpy.asarray(vvop[:,a0:a1], order='C')
            cache_col_a_bar = numpy.zeros_like(cache_col_a)
        contract(a0, a1, a0, a1,
                (cache_row_a, cache_col_a, cache_row_a, cache_col_a),
                (cache_row_a_bar, cache_col_a_bar, cache_row_a_bar, cache_col_a_bar))

        for b0, b1 in prange(0, a0, bufsize//4):
            cache_row_b = numpy.asarray(vvop[b0:b1,:], order='C')
            cache_row_b_bar = numpy.zeros_like(cache_row_b)
            cache_col_b = numpy.asarray(vvop[:,b0:b1], order='C')
            cache_col_b_bar = numpy.zeros_like(cache_col_b)
            contract(a0, a1, b0, b1,
                    (cache_row_a, cache_col_a, cache_row_b, cache_col_b),
                    (cache_row_a_bar, cache_col_a_bar, cache_row_b_bar, cache_col_b_bar))

            vvop_bar[b0:b1,:] += cache_row_b_bar
            vvop_bar[:,b0:b1] += cache_col_b_bar

        vvop_bar[a0:a1,:] += cache_row_a_bar
        if (a0, a1) != (0, nvir):
            vvop_bar[:,a0:a1] += cache_col_a_bar

    ovoo_bar = numpy.asarray(vooo_bar.transpose(1,0,3,2))
    ovov_bar = numpy.asarray(vvop_bar[:,:,:,:nocc].transpose(2,0,3,1))
    ovvv_bar = vvop_bar[:,:,:,nocc:].transpose(2,0,3,1)
    ovvv_bar += ovvv_bar.transpose(0,1,3,2)
    idx, idy = numpy.diag_indices(nvir)
    ovvv_bar[:,:,idx,idy] *= .5
    idx, idy = numpy.tril_indices(nvir)
    ovvv_tril_bar = numpy.asarray(ovvv_bar[:,:,idx,idy])

    mytimer.timer('_ccsd_t_energy_bwd:')
    return mat_bar, t1T_bar, t2T_bar, mo_energy_bar, fvo_bar, ovoo_bar, ovov_bar, ovvv_tril_bar

_ccsd_t_energy.defvjp(_ccsd_t_energy_fwd, _ccsd_t_energy_bwd)
