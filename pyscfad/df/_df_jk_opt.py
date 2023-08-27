from functools import partial
import ctypes
import numpy
from jax import custom_vjp
from jax.tree_util import tree_flatten, tree_unflatten
from pyscf import lib
from pyscf.lib import logger
from pyscf.df import df_jk as pyscf_df_jk
from pyscfadlib import libcvhf_vjp as libvhf

libao2mo = lib.load_library('libao2mo')

@partial(custom_vjp, nondiff_argnums=(2,3,4,5))
def get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13):
    return pyscf_df_jk.get_jk(dfobj, dm, hermi=hermi,
                              with_j=with_j, with_k=with_k,
                              direct_scf_tol=direct_scf_tol)

def get_jk_fwd(dfobj, dm, hermi, with_j, with_k, direct_scf_tol):
    vj, vk = get_jk(dfobj, dm, hermi, with_j, with_k, direct_scf_tol)
    return (vj, vk), (dfobj, dm)

def get_jk_bwd(hermi, with_j, with_k, direct_scf_tol,
               res, ybar):
    dfobj, dm = res
    vj_bar, vk_bar = ybar

    log = logger.new_logger(dfobj)
    fmmm = libao2mo.AO2MOmmm_bra_nr_s2
    fdrv = libao2mo.AO2MOnr_e2_drv
    ftrans = libao2mo.AO2MOtranse2_nr_s2
    vjpdrv = libvhf.df_vk_vjp
    null = lib.c_null_ptr()

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]

    vj_bar = numpy.asarray(vj_bar).reshape(-1,nao,nao)
    if with_j:
        idx = numpy.arange(nao)
        dmtril = lib.pack_tril(dms + dms.conj().transpose(0,2,1))
        dmtril[:,idx*(idx+1)//2+idx] *= .5

        vj_bar_tril = lib.pack_tril(vj_bar + vj_bar.conj().transpose(0,2,1))
        vj_bar_tril[:,idx*(idx+1)//2+idx] *= .5

    #TODO save eri_bar on disk
    eri_bar = numpy.zeros((dfobj.get_naoaux(), nao*(nao+1)//2))
    dms_bar = [numpy.zeros((nao,nao), order='F'),] * nset

    vk_bar = numpy.asarray(vk_bar).reshape(-1,nao,nao)
    vk_bar = [numpy.asarray(x, order='F') for x in vk_bar]
    dms = [numpy.asarray(x, order='F') for x in dms]

    rargs = (ctypes.c_int(nao), (ctypes.c_int*4)(0, nao, 0, nao),
             null, ctypes.c_int(0))
    max_memory = dfobj.max_memory - lib.current_memory()[0]
    blksize = max(4, int(min(dfobj.blockdim, max_memory*.4e6/8/nao**2)))
    buf = numpy.empty((blksize,nao,nao))
    p1 = 0
    for eri1 in dfobj.loop(blksize):
        naux, nao_pair = eri1.shape
        p0, p1 = p1, p1 + naux
        if with_j:
            rho_bar = numpy.einsum('ix,px->ip', vj_bar_tril, eri1)
            dmtril_bar = numpy.einsum('ip,px->ix', rho_bar, eri1)
            dms_bar += lib.unpack_tril(dmtril_bar)

            rho = numpy.einsum('ix,px->ip', dmtril, eri1)
            eri_bar[p0:p1] += numpy.einsum('ip,ix->px', rho, vj_bar_tril)
            eri_bar[p0:p1] += numpy.einsum('ix,ip->px', dmtril, rho_bar)

        for k in range(nset):
            #TODO save buf1 on disk to avoid recomputation
            buf1 = buf[:naux]
            fdrv(ftrans, fmmm,
                 buf1.ctypes.data_as(ctypes.c_void_p),
                 eri1.ctypes.data_as(ctypes.c_void_p),
                 dms[k].ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux), *rargs)

            vjpdrv(eri_bar[p0:p1].ctypes.data_as(ctypes.c_void_p),
                   dms_bar[k].ctypes.data_as(ctypes.c_void_p),
                   vk_bar[k].ctypes.data_as(ctypes.c_void_p),
                   buf1.ctypes.data_as(ctypes.c_void_p),
                   eri1.ctypes.data_as(ctypes.c_void_p),
                   dms[k].ctypes.data_as(ctypes.c_void_p),
                   ctypes.c_int(naux), ctypes.c_int(nao))

    dm_bar = numpy.asarray(dms_bar).reshape(dm_shape)
    #TODO need a better way to add vjps for objects
    leaves, tree = tree_flatten(dfobj)
    shapes = [leaf.shape for leaf in leaves]
    leaves = [numpy.zeros(shape) for shape in shapes[:-1]]
    leaves.append(eri_bar)
    dfobj_bar = tree_unflatten(tree, leaves)
    log.timer('get_jk_bwd')
    del log
    return (dfobj_bar, dm_bar)

get_jk.defvjp(get_jk_fwd, get_jk_bwd)
