from functools import partial
import ctypes
import numpy
from jax import custom_vjp
from pyscf.ao2mo import _ao2mo
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
    out = nr_e2(eri, mo_coeff, orbs_slice, aosym=aosym, mosym=mosym,
                out=out, ao_loc=ao_loc)
    return out, (eri, mo_coeff)


def nr_e2_bwd(orbs_slice, aosym, mosym, out, ao_loc,
              res, ybar):
    eri, mo_coeff = res
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
    eri_bar = numpy.zeros_like(eri, order='C')

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

nr_e2.defvjp(nr_e2_fwd, nr_e2_bwd)
