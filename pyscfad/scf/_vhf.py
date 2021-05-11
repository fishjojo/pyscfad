from functools import partial
import numpy
from jax import custom_jvp
from pyscf.scf import _vhf

@partial(custom_jvp, nondiff_argnums=(2,3,4))
def incore(eri, dms, hermi=0, with_j=True, with_k=True):
    eri = numpy.asarray(eri)
    dms = numpy.asarray(dms)
    return _vhf.incore(eri, dms, hermi, with_j, with_k)

@incore.defjvp
def incore_jvp(hermi, with_j, with_k,
               primals, tangents):
    raise NotImplementedError
    #eri, dms, = primals
    #eri_t, dms_t, = tangents

    #vj, vk = incore(eri, dms, hermi, with_j, with_k)

    #nao = dms_shape[-1]
    #npair = nao*(nao+1)//2
    #if eri.ndim == 2 and npair*npair == eri.size:
    #    pass
    #elif eri.ndim == 1 and npair*(npair+1)//2 == eri.size:
    #    pass
    #else:
    #    raise RuntimeError

    #vj_dot = vk_dot = None
    #return (vj, vk), (vj_dot, vk_dot)
