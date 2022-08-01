from functools import reduce
from jax import vmap
from pyscf import numpy as np
from pyscf.scf import diis as pyscf_cdiis
from pyscfad.lib import jit
from pyscfad.lib import diis

class CDIIS(diis.DIIS, pyscf_cdiis.CDIIS):
    def __init__(self, mf=None, filename=None):
        pyscf_cdiis.CDIIS.__init__(self, mf=mf, filename=filename)
        self.incore = True

@jit
def get_err_vec(s, d, f):
    '''error vector = SDF - FDS'''
    f = np.asarray(f)
    s = np.asarray(s).reshape(f.shape)
    d = np.asarray(d).reshape(f.shape)

    def body(s, d, f):
        sdf = reduce(np.dot, (s,d,f))
        return sdf.T.conj() - sdf

    if f.ndim == 2:
        errvec = body(s, d, f)
    else:
        errvec = vmap(body)(s, d, f)
    return errvec

SCFDIIS = SCF_DIIS = DIIS = CDIIS
