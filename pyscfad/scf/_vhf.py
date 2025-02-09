from functools import partial
import ctypes
import numpy
from jax import custom_vjp
from pyscf.scf import _vhf
from pyscfadlib import libcvhf_vjp as libcvhf

def incore(eri, dms, hermi=0, with_j=True, with_k=True):
    vj = vk = None
    if with_j:
        vj = _get_j(eri, dms, hermi=hermi)
    if with_k:
        vk = _get_k(eri, dms, hermi=hermi)
    return vj, vk

@partial(custom_vjp, nondiff_argnums=(2,))
def _get_j(eri, dms, hermi=0):
    nao = dms.shape[-1]
    npair = nao*(nao+1)//2
    if eri.ndim == 2 and eri.size == npair*npair:
        vj, _ = _vhf.incore(eri, dms, hermi=hermi,
                             with_j=True, with_k=False)
    else:
        raise NotImplementedError
    return vj

def _get_j_fwd(eri, dms, hermi):
    return _get_j(eri, dms, hermi), (eri, dms)

def _get_j_bwd(hermi, res, vjk_bar):
    #t0 = (logger.process_clock(), logger.perf_counter())
    eri, dms = res
    eri = numpy.asarray(eri, order='C', dtype=numpy.double)
    dms = numpy.asarray(dms, order='C', dtype=numpy.double)

    dms_shape = dms.shape
    nao = dms_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    ndm = dms.shape[0]

    vjk_bar = numpy.asarray(vjk_bar, order='C', dtype=numpy.double)
    vjk_bar = vjk_bar.reshape(dms.shape)

    vjp_dms = numpy.zeros_like(dms, order='C', dtype=numpy.double)
    fdrv = getattr(libcvhf, 'CVHFnrs4_incore_dms_vjp')
    fvjk = getattr(libcvhf, 'CVHFics4_vj_dms_deriv')

    fdrv(vjp_dms.ctypes.data_as(ctypes.c_void_p),
         eri.ctypes.data_as(ctypes.c_void_p),
         vjk_bar.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(ndm), ctypes.c_int(nao), fvjk)
    vjp_dms = vjp_dms.reshape(dms_shape)

    vjp_eri = numpy.zeros_like(eri, order='C', dtype=numpy.double)
    fdrv = getattr(libcvhf, 'CVHFnrs4_incore_eri_vjp')
    fvjk = getattr(libcvhf, 'CVHFics4_vj_eri_deriv')
    fdrv(vjp_eri.ctypes.data_as(ctypes.c_void_p),
         dms.ctypes.data_as(ctypes.c_void_p),
         vjk_bar.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(ndm), ctypes.c_int(nao), fvjk)
    #t1 = (logger.process_clock(), logger.perf_counter())
    #print('    CPU time for %s %9.2f sec, wall time %9.2f sec'
    #              % ('_get_j_bwd', t1[0]-t0[0], t1[1]-t0[1]))
    return vjp_eri, vjp_dms

_get_j.defvjp(_get_j_fwd, _get_j_bwd)


@partial(custom_vjp, nondiff_argnums=(2,))
def _get_k(eri, dms, hermi=0):
    nao = dms.shape[-1]
    npair = nao*(nao+1)//2
    if eri.ndim == 2 and eri.size == npair*npair:
        _, vk = _vhf.incore(eri, dms, hermi=hermi,
                             with_j=False, with_k=True)
    else:
        raise NotImplementedError
    return vk

def _get_k_fwd(eri, dms, hermi):
    return _get_k(eri, dms, hermi), (eri, dms)

def _get_k_bwd(hermi, res, vjk_bar):
    #t0 = (logger.process_clock(), logger.perf_counter())
    eri, dms = res
    eri = numpy.asarray(eri, order='C', dtype=numpy.double)
    dms = numpy.asarray(dms, order='C', dtype=numpy.double)

    dms_shape = dms.shape
    nao = dms_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    ndm = dms.shape[0]

    vjk_bar = vjk_bar.reshape(dms.shape)
    if hermi != 0:
        tmp = .5 * (vjk_bar + vjk_bar.transpose(0,2,1))
    vjk_bar = numpy.asarray(vjk_bar, order='C', dtype=numpy.double)

    vjp_dms = numpy.zeros_like(dms, order='C', dtype=numpy.double)
    fdrv = getattr(libcvhf, 'CVHFnrs4_incore_dms_vjp')
    fvjk = getattr(libcvhf, 'CVHFics4_vk_dms_deriv')

    fdrv(vjp_dms.ctypes.data_as(ctypes.c_void_p),
         eri.ctypes.data_as(ctypes.c_void_p),
         vjk_bar.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(ndm), ctypes.c_int(nao), fvjk)
    vjp_dms = vjp_dms.reshape(dms_shape)

    vjp_eri = numpy.zeros_like(eri, order='C', dtype=numpy.double)
    fdrv = getattr(libcvhf, 'CVHFnrs4_incore_eri_vjp')
    fvjk = getattr(libcvhf, 'CVHFics4_vk_eri_deriv')
    fdrv(vjp_eri.ctypes.data_as(ctypes.c_void_p),
         dms.ctypes.data_as(ctypes.c_void_p),
         vjk_bar.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(ndm), ctypes.c_int(nao), fvjk)
    #t1 = (logger.process_clock(), logger.perf_counter())
    #print('    CPU time for %s %9.2f sec, wall time %9.2f sec'
    #              % ('_get_k_bwd', t1[0]-t0[0], t1[1]-t0[1]))
    return vjp_eri, vjp_dms

_get_k.defvjp(_get_k_fwd, _get_k_bwd)
