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
from jax import custom_vjp
from jax.custom_derivatives import SymbolicZero
from pyscf.ao2mo import _ao2mo
from pyscfad.lib import _cotangent_buffer
from pyscfadlib import libao2mo_vjp as libao2mo

def _fpointer(name):
    return getattr(libao2mo, name)

@partial(custom_vjp, nondiff_argnums=(2,3,4,5,6))
def nr_e2(eri, mo_coeff, orbs_slice, aosym='s1', mosym='s1', out=None,
          ao_loc=None):
    eri = numpy.asarray(eri, order='C')
    mo_coeff = numpy.asarray(mo_coeff, order='F')
    return _ao2mo.nr_e2(eri, mo_coeff, orbs_slice,
                        aosym=aosym, mosym=mosym,
                        out=out, ao_loc=ao_loc)

def nr_e2_fwd(eri, mo_coeff, orbs_slice, aosym, mosym, out, ao_loc):
    perturbed = (eri.perturbed, mo_coeff.perturbed)
    eri = eri.value
    mo_coeff = mo_coeff.value
    out = nr_e2(eri, mo_coeff, orbs_slice, aosym=aosym, mosym=mosym,
                out=out, ao_loc=ao_loc)
    return out, (eri, mo_coeff, perturbed)

def nr_e2_bwd(orbs_slice, aosym, mosym, out, ao_loc,
              res, ybar):
    eri, mo_coeff, (eri_perturbed, mo_perturbed) = res
    if isinstance(ybar, SymbolicZero):
        return (None, None)
    # added in place to the cotangent buffer of eri if registered
    eri_bar_buf = _cotangent_buffer.get(eri) if eri_perturbed else None
    eri_bar, mo_coeff_bar = nr_e2_vjp(eri, mo_coeff, orbs_slice, ybar,
                                      aosym=aosym, mosym=mosym,
                                      ao_loc=ao_loc, eri_bar=eri_bar_buf)
    if eri_bar_buf is not None or not eri_perturbed:
        eri_bar = None
    if not mo_perturbed:
        mo_coeff_bar = None
    return (eri_bar, mo_coeff_bar)

def nr_e2_vjp(eri, mo_coeff, orbs_slice, ybar, aosym='s1', mosym='s1',
              ao_loc=None, eri_bar=None):
    '''Cotangents of eri (added to eri_bar if given) and mo_coeff
    for the cotangent ybar of nr_e2.'''
    nrow, nao_pair = eri.shape
    nao, nmo = mo_coeff.shape
    assert nrow == ybar.shape[0]

    k0, k1, l0, l1 = orbs_slice
    kc = k1 - k0
    lc = l1 - l0
    kl_count = kc * lc
    assert kl_count == ybar.shape[1]

    if aosym in ('s4', 's2', 's2kl'):
        assert nao_pair == nao*(nao+1)// 2
        if mosym == 's2':
            raise NotImplementedError(f'nr_e2_bwd: mosym={mosym} not supported')
        elif kc <= lc:
            fmmm = _fpointer('AO2MOmmm_nr_vjp_s2_iltj')
            mo_coeff = numpy.asarray(mo_coeff, order='F')
            mo_coeff_bar = numpy.zeros_like(mo_coeff, order='C')
        else:
            fmmm = _fpointer('AO2MOmmm_nr_vjp_s2_igtj')
            mo_coeff = numpy.asarray(mo_coeff, order='C')
            mo_coeff_bar = numpy.zeros_like(mo_coeff, order='F')
    else:
        raise NotImplementedError(f'nr_e2_bwd: aosym={aosym} not supported')

    if ao_loc is None:
        ftrans = _fpointer('AO2MOtranse2_nr_vjp_' + aosym)
    else:
        raise NotImplementedError

    eri = numpy.asarray(eri, order='C')
    ybar = numpy.asarray(ybar, order='C')

    #TODO save eri_bar on disk
    if eri_bar is None:
        eri_bar = numpy.zeros_like(eri, order='C')
    else:
        assert eri_bar.shape == eri.shape and eri_bar.flags.c_contiguous

    fdrv = getattr(libao2mo, 'AO2MOnr_e2_vjp_drv')
    fdrv(ftrans, fmmm,
         eri_bar.ctypes.data_as(ctypes.c_void_p),
         mo_coeff_bar.ctypes.data_as(ctypes.c_void_p),
         eri.ctypes.data_as(ctypes.c_void_p),
         mo_coeff.ctypes.data_as(ctypes.c_void_p),
         ybar.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(nrow), ctypes.c_int(nao), ctypes.c_int(nmo),
         (ctypes.c_int*4)(*orbs_slice))
    return (eri_bar, mo_coeff_bar)

nr_e2.defvjp(nr_e2_fwd, nr_e2_bwd, symbolic_zeros=True)
