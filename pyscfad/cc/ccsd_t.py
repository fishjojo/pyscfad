import ctypes
import numpy
from jax import custom_vjp
from jax.tree_util import tree_flatten, tree_unflatten
from pyscf.lib import (
    prange_tril,
    num_threads,
    current_memory,
#    load_library
)
from pyscf.cc import ccsd_t as pyscf_ccsd_t
from pyscfad.lib import logger
#libcc = load_library('libcc')
from pyscfadlib import libcc_vjp as libcc

def kernel(mycc, eris, t1=None, t2=None, verbose=logger.NOTE):
    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2

    max_memory = mycc.max_memory

    @custom_vjp
    def _ccsd_t_kernel(eris, t1, t2):
        t1 = numpy.asarray(t1)
        t2 = numpy.asarray(t2, order='C')
        eris.fock = numpy.asarray(eris.fock, order='C')
        et = pyscf_ccsd_t.kernel(mycc, eris, t1, t2, verbose)
        return et

    def _ccsd_t_kernel_fwd(eris, t1, t2):
        et = _ccsd_t_kernel(eris, t1, t2)
        return et, (eris, t1, t2)

    def _ccsd_t_kernel_bwd(res, ybar):
        eris, t1, t2 = res

        leaves, tree = tree_flatten(eris)
        assert len(leaves) == 9
        shapes = [leaf.shape for leaf in leaves]
        del leaves

        t1_bar, t2_bar, fock_bar, mo_energy_bar,\
            ovoo_bar, ovov_bar, ovvv_bar = _ccsd_t_energy_vjp(eris, t1, t2, ybar, max_memory)

        leaves = [fock_bar,
                  mo_energy_bar,
                  numpy.zeros(shapes[2]),
                  ovoo_bar,
                  ovov_bar,
                  numpy.zeros(shapes[5]),
                  numpy.zeros(shapes[6]),
                  ovvv_bar,
                  numpy.zeros(shapes[8])]
        eris_bar = tree_unflatten(tree, leaves)
        return eris_bar, t1_bar, t2_bar

    _ccsd_t_kernel.defvjp(_ccsd_t_kernel_fwd, _ccsd_t_kernel_bwd)
    return _ccsd_t_kernel(eris, t1, t2)

def _ccsd_t_energy_vjp(eris, t1, t2, et_bar, max_memory):
    nocc, nvir = t1.shape
    nmo = nocc + nvir

    et_bar *= 2

    t1T = numpy.asarray(t1.T, order='C')
    t1T_bar = numpy.zeros_like(t1T)
    t2T = numpy.asarray(t2.transpose(2,3,1,0), order='C')
    t2T_bar = numpy.zeros_like(t2T)

    mo_energy = numpy.asarray(eris.mo_energy, order='C')
    mo_energy_bar = numpy.zeros_like(mo_energy)
    fvo = numpy.asarray(eris.fock[nocc:,:nocc], order='C')
    fvo_bar = numpy.zeros_like(fvo)

    vooo = numpy.asarray(eris.ovoo).conj().transpose(1,0,3,2)
    vooo = numpy.asarray(vooo, order='C')
    vooo_bar = numpy.zeros_like(vooo)

    vvop = numpy.empty((nvir,nvir,nocc,nmo))
    vvop[:,:,:,:nocc] = numpy.asarray(eris.ovov).conj().transpose(1,3,0,2)
    vvop[:,:,:,nocc:] = eris.get_ovvv().conj().transpose(1,3,0,2)
    vvop = numpy.asarray(vvop, order='C')
    vvop_bar = numpy.zeros_like(vvop)

    drv = libcc.ccsd_t_energy_vjp
    def contract(a0, a1, b0, b1, cache, cache_bar):
        cache_row_a, cache_col_a, cache_row_b, cache_col_b = cache
        cache_row_a_bar, cache_col_a_bar, cache_row_b_bar, cache_col_b_bar = cache_bar
        drv(mo_energy.ctypes.data_as(ctypes.c_void_p),
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
    min_memory = (nvir**2*nocc**2+nvir*nocc**3+nocc**3*6+2*nvir*nocc+nmo)*num_threads()*8/1e6
    if max_memory < min_memory:
        raise RuntimeError(f'_ccsd_t_energy_vjp: at least {min_memory} MB of memory needed.')
    bufsize = (max_memory - min_memory)*1e6/8/num_threads()/(nocc*nmo)
    bufsize *= .5
    bufsize *= .8
    bufsize = max(8, bufsize)

    for a0, a1 in reversed(list(prange_tril(0, nvir, bufsize))):
        cache_row_a = numpy.asarray(vvop[a0:a1,:a1], order='C')
        cache_row_a_bar = numpy.zeros_like(cache_row_a)
        if a0 == 0:
            cache_col_a = cache_row_a
            cache_col_a_bar = cache_row_a_bar
        else:
            cache_col_a = numpy.asarray(vvop[:a0,a0:a1], order='C')
            cache_col_a_bar = numpy.zeros_like(cache_col_a)
        contract(a0, a1, a0, a1,
                (cache_row_a, cache_col_a, cache_row_a, cache_col_a),
                (cache_row_a_bar, cache_col_a_bar, cache_row_a_bar, cache_col_a_bar))

        for b0, b1 in prange_tril(0, a0, bufsize/4):
            cache_row_b = numpy.asarray(vvop[b0:b1,:b1], order='C')
            cache_row_b_bar = numpy.zeros_like(cache_row_b)
            if b0 == 0:
                cache_col_b = cache_row_b
                cache_col_b_bar = cache_row_b_bar
            else:
                cache_col_b = numpy.asarray(vvop[:b0,b0:b1], order='C')
                cache_col_b_bar = numpy.zeros_like(cache_col_b)
            contract(a0, a1, b0, b1,
                    (cache_row_a, cache_col_a, cache_row_b, cache_col_b),
                    (cache_row_a_bar, cache_col_a_bar, cache_row_b_bar, cache_col_b_bar))

            vvop_bar[b0:b1,:b1] += cache_row_b_bar
            if b0 != 0:
                vvop_bar[:b0,b0:b1] += cache_col_b_bar

        vvop_bar[a0:a1,:a1] += cache_row_a_bar
        if a0 != 0:
            vvop_bar[:a0,a0:a1] += cache_col_a_bar

    t1_bar = numpy.asarray(t1T_bar.T)
    t2_bar = numpy.asarray(t2T_bar.transpose(3,2,0,1))
    fock_bar = numpy.zeros((nmo,nmo))
    fock_bar[nocc:,:nocc] = fvo_bar

    ovoo_bar = numpy.asarray(vooo_bar.transpose(1,0,3,2))
    ovov_bar = numpy.asarray(vvop_bar[:,:,:,:nocc].transpose(2,0,3,1))
    ovvv_bar = vvop_bar[:,:,:,nocc:].transpose(2,0,3,1)
    ovvv_bar += ovvv_bar.transpose(0,1,3,2)
    idx, idy = numpy.diag_indices(nvir)
    ovvv_bar[:,:,idx,idy] *= .5
    idx, idy = numpy.tril_indices(nvir)
    ovvv_tril_bar = numpy.asarray(ovvv_bar[:,:,idx,idy])
    return t1_bar, t2_bar, fock_bar, mo_energy_bar, ovoo_bar, ovov_bar, ovvv_tril_bar
